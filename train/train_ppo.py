"""
train_ppo.py

Entrypoint for offline multi-turn PPO training with verl.

This follows verl's RayPPOTrainer pattern but:
  - Skips the online rollout worker (data is pre-collected)
  - Uses MultiTurnOfflineDataset for loading parquet files
  - Uses OfflineMultiTurnRewardManager to pass pre-computed rewards through
  - Configures GAE with γ=0.9 and λ=0.95 for noisy per-turn rewards

Usage:
  python train_ppo.py \
    --config config/ppo_multiturn.yaml

Or with Ray:
  ray start --head
  python train_ppo.py --config config/ppo_multiturn.yaml
"""

import os
import ray
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

import verl.utils.hdfs_io as hdfs_io
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

from multiturn_dataset import MultiTurnOfflineDataset, OfflineMultiTurnRewardManager


# ---------------------------------------------------------------------------
# Offline-aware subclass of RayPPOTrainer
# ---------------------------------------------------------------------------


class OfflineMultiTurnPPOTrainer(RayPPOTrainer):
    """
    Extends RayPPOTrainer for offline multi-turn data.

    Key differences:
      1. _build_dataloader: uses MultiTurnOfflineDataset instead of RLHFDataset
      2. _build_reward_fn: uses OfflineMultiTurnRewardManager
      3. fit loop: skips generate_sequences() — responses are already in the batch

    GAE is handled by the parent class's compute_advantage() with our γ/λ config.
    """

    def _build_dataloader(self):
        config = self.config
        tokenizer = self.tokenizer

        train_dataset = MultiTurnOfflineDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            max_len=config.data.max_length,
        )

        # verl expects a DataLoader that yields dict batches
        from torch.utils.data import DataLoader
        from multiturn_dataset import ConversationGroupedBatchSampler

        def collate_fn(batch):
            # Pad 'responses' to the same length within the batch
            max_resp_len = max(b["responses"].shape[0] for b in batch)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            collated = {}
            for key in batch[0]:
                if key == "responses":
                    padded = torch.stack(
                        [
                            torch.cat(
                                [
                                    b[key],
                                    torch.full(
                                        (max_resp_len - b[key].shape[0],), pad_id, dtype=torch.long
                                    ),
                                ]
                            )
                            for b in batch
                        ]
                    )
                    collated[key] = padded
                else:
                    collated[key] = torch.stack([b[key] for b in batch])
            return collated

        # Use conversation-grouped sampler to prevent a conversation and its
        # prefixes from appearing in the same batch (would double-count gradients)
        batch_sampler = ConversationGroupedBatchSampler(
            train_dataset,
            batch_size=config.trainer.train_batch_size,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        return train_dataloader

    def _build_reward_fn(self):
        return OfflineMultiTurnRewardManager(
            tokenizer=self.tokenizer,
            num_examine=self.config.reward_model.get("num_examine", 2),
            normalize_per_sample=True,
            clip_rewards=True,
        )

    def fit(self):
        """
        Offline PPO loop.

        Unlike the online loop, we skip generate_sequences() and instead
        use the pre-tokenized responses already present in the batch.
        """
        from verl.utils.tracking import Tracking
        from tqdm import tqdm

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        train_dataloader = self._build_dataloader()
        reward_fn = self._build_reward_fn()

        global_step = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                metrics = {}

                # Wrap in DataProto
                batch = DataProto.from_single_dict(batch_dict)

                # -------------------------------------------------------
                # Step 1: Compute reference log-probs (KL regularization)
                # -------------------------------------------------------
                if self.use_reference_policy:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                # -------------------------------------------------------
                # Step 2: Compute values (critic forward pass)
                # -------------------------------------------------------
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

                # -------------------------------------------------------
                # Step 3: Apply pre-computed rewards + KL penalty
                # -------------------------------------------------------
                reward_result = reward_fn(batch, return_dict=True)
                batch.batch["token_level_rewards"] = reward_result["reward_tensor"]
                metrics.update(
                    {
                        f"reward/{k}": v.item() if hasattr(v, "item") else v
                        for k, v in reward_result.get("reward_extra_info", {}).items()
                    }
                )

                # Apply KL penalty to token-level rewards
                batch, kl_metrics = self.apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)

                # -------------------------------------------------------
                # Step 4: Compute advantages via GAE
                # (γ and λ come from config — see ppo_multiturn.yaml)
                # -------------------------------------------------------
                batch = self.compute_advantage(batch)

                # -------------------------------------------------------
                # Step 5: Update actor and critic
                # -------------------------------------------------------
                actor_metrics = self.actor_rollout_wg.update_actor(batch)
                critic_metrics = self.critic_wg.update_critic(batch)

                metrics.update(actor_metrics.non_tensor_batch.get("metrics", {}))
                metrics.update(critic_metrics.non_tensor_batch.get("metrics", {}))

                # -------------------------------------------------------
                # Logging & checkpointing
                # -------------------------------------------------------
                logger.log(data=metrics, step=global_step)

                if global_step % self.config.trainer.save_freq == 0 and global_step > 0:
                    self._save_checkpoint(global_step)

                global_step += 1

        logger.finish()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="ppo_multiturn", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}},
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Worker resource pools
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=1,
    )

    # Actor + reference policy (colocated to save memory on smaller setups)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ActorRolloutRefWorker,
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )

    critic_cls = RayClassWithInitArgs(
        cls=CriticWorker,
        config=config.critic,
    )

    all_wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=create_colocated_worker_cls(
            {
                "actor_rollout": actor_rollout_cls,
                "critic": critic_cls,
            }
        ),
    )

    # Reference policy worker (separate if memory allows, else colocated above)
    ref_policy_cls = RayClassWithInitArgs(
        cls=ActorRolloutRefWorker,
        config=config.actor_rollout_ref,
        role="ref",
    )
    ref_wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ref_policy_cls,
    )

    trainer = OfflineMultiTurnPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        actor_rollout_wg=all_wg,
        critic_wg=all_wg,
        ref_policy_wg=ref_wg,
        train_reward_fn=None,  # handled inside fit()
        val_reward_fn=None,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
