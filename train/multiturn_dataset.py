"""
multiturn_dataset.py

Custom verl Dataset and RewardManager for offline multi-turn RL training.

The dataset loads pre-tokenized parquet files (produced by prepare_data.py)
and wraps them in verl's DataProto format.

The reward manager simply reads the pre-computed token_level_rewards that
were baked in during preprocessing — no runtime reward computation needed.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MultiTurnOfflineDataset(Dataset):
    """
    Loads a parquet file produced by prepare_data.py.

    Each row becomes one training sample. verl's DataLoader will collate
    these into batched DataProto objects.

    Padding is applied here to a fixed max_len so batches are rectangular.
    """

    def __init__(
        self,
        data_files: list[str] | str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 4096,
    ):
        if isinstance(data_files, str):
            data_files = [data_files]

        dfs = [pd.read_parquet(f) for f in data_files]
        self.df = pd.concat(dfs, ignore_index=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        print(f"[MultiTurnOfflineDataset] Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        def pad_or_trunc(arr: list, pad_value: int | float, dtype) -> torch.Tensor:
            t = torch.tensor(arr, dtype=dtype)
            length = t.shape[0]
            if length >= self.max_len:
                return t[: self.max_len]
            pad_size = self.max_len - length
            pad = torch.full((pad_size,), pad_value, dtype=dtype)
            return torch.cat([t, pad], dim=0)

        input_ids = pad_or_trunc(row["input_ids"], self.pad_id, torch.long)
        attention_mask = pad_or_trunc(row["attention_mask"], 0, torch.long)
        position_ids = pad_or_trunc(row["position_ids"], 0, torch.long)
        loss_mask = pad_or_trunc(row["loss_mask"], 0, torch.long)
        response_mask = pad_or_trunc(row["response_mask"], 0, torch.long)
        token_level_rewards = pad_or_trunc(row["token_level_rewards"], 0.0, torch.float32)

        # responses: concatenated assistant tokens (variable length, no padding needed here
        # as verl's actor/critic workers don't require a fixed size for this field)
        responses = torch.tensor(row["responses"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
            "responses": responses,
        }


# ---------------------------------------------------------------------------
# Batch Sampler: prevents a conversation and its prefixes from co-occurring
# ---------------------------------------------------------------------------


class ConversationGroupedBatchSampler(Sampler):
    """
    Yields batches where no two samples share the same conversation_id.

    This prevents a conversation and its prefix augmentations from appearing
    in the same PPO minibatch, which would double-count gradient signal on
    shared early turns and destabilize training.

    Strategy:
      - Group all sample indices by conversation_id
      - Each batch is assembled by picking ONE sample per conversation_id
        (randomly chosen from that conversation's available samples)
      - Shuffle conversation order each epoch

    Usage:
        sampler = ConversationGroupedBatchSampler(dataset, batch_size=32)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=...)
    """

    def __init__(self, dataset: MultiTurnOfflineDataset, batch_size: int, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)

        # Group indices by conversation_id
        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            cid = int(dataset.df.iloc[idx]["conversation_id"])
            self.groups[cid].append(idx)
        self.conversation_ids = list(self.groups.keys())

    def __iter__(self):
        # Shuffle conversations each epoch
        conv_ids = self.conversation_ids.copy()
        self.rng.shuffle(conv_ids)

        batch = []
        for cid in conv_ids:
            # Pick one sample from this conversation (full or any prefix)
            chosen = self.rng.choice(self.groups[cid])
            batch.append(chosen)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Drop the last incomplete batch (drop_last=True equivalent)

    def __len__(self):
        return len(self.conversation_ids) // self.batch_size


# ---------------------------------------------------------------------------
# Reward Manager
# ---------------------------------------------------------------------------


class OfflineMultiTurnRewardManager(AbstractRewardManager):
    """
    Reward manager for offline multi-turn training.

    Rewards were pre-computed during preprocessing and stored in the
    `token_level_rewards` field of each DataProto batch. This manager
    simply passes them through — no runtime LLM judge calls needed.

    It also applies:
      - Per-conversation reward normalization (z-score) to correct for
        LLM judge calibration drift between conversations.
      - Optional reward clipping to [-1, 1].
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        normalize_per_sample: bool = True,
        clip_rewards: bool = True,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_per_sample = normalize_per_sample
        self.clip_rewards = clip_rewards

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = data.batch["token_level_rewards"].clone()  # (batch, seq_len)
        response_mask = data.batch["response_mask"].float()  # (batch, seq_len)

        # Zero out rewards on non-response tokens (sanity check)
        reward_tensor = reward_tensor * response_mask

        if self.normalize_per_sample:
            reward_tensor = self._normalize_per_sample(reward_tensor, response_mask)

        if self.clip_rewards:
            reward_tensor = torch.clamp(reward_tensor, -1.0, 1.0)

        if self.num_examine > 0:
            self._log_samples(data, reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "mean_reward": reward_tensor.sum() / (response_mask.sum() + 1e-8),
                },
            }
        return reward_tensor

    def _normalize_per_sample(
        self,
        reward_tensor: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Z-score normalize rewards within each conversation independently.

        This makes the signal relative (which turn was better/worse within
        this conversation) rather than absolute, which is more robust to
        LLM judge calibration drift.

        Only normalizes over positions that actually have a reward signal
        (i.e., the last token of each assistant turn — non-zero reward positions).
        """
        normalized = reward_tensor.clone()

        for i in range(reward_tensor.shape[0]):
            # Find positions with non-zero reward signal
            reward_pos = reward_tensor[i].abs() > 1e-6
            if reward_pos.sum() < 2:
                # Only 1 reward in this conversation — can't normalize
                continue

            vals = reward_tensor[i][reward_pos]
            mean = vals.mean()
            std = vals.std()
            if std < 1e-8:
                # All rewards identical — centering only
                normalized[i][reward_pos] = vals - mean
            else:
                normalized[i][reward_pos] = (vals - mean) / (std + 1e-8)

        return normalized

    def _log_samples(self, data: DataProto, reward_tensor: torch.Tensor):
        """Print a few samples for debugging during training."""
        n = min(self.num_examine, reward_tensor.shape[0])
        for i in range(n):
            ids = data.batch["input_ids"][i]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            rewards = reward_tensor[i][reward_tensor[i].abs() > 1e-6].tolist()
            print(f"\n[RewardManager] Sample {i}")
            print(f"  text (truncated): {text[:200]}...")
            print(f"  per-turn rewards: {[f'{r:.3f}' for r in rewards]}")
