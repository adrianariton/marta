import uuid
import uuid
from pathlib import Path
from datetime import datetime
from core.attacks.interfaces import MessageLogger, MessageType, MsgFormat
from core.attacks.interfaces import Tag, Conversation
from typing import Callable, Optional, Literal
from core.attacks.agents import AutoAgent, DummyAgent
from core.attacks.datastore.yielder import _Yielder
import numpy as np
import pandas as pd
from core.attacks.datastore.train.utils import ReplayBuffer, sanitize_reward, filter_dataset
from core.attacks.datastore.train.rewardtrainer import TrainerWrapper
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/example.log", encoding="utf-8", level=logging.DEBUG)


class RewardBatchTrainerLogger(MessageLogger):
    """Expected message timeline:
    [
        INPUT, EVAL, IS_REFUSED(True)
        |
        INPUT, EVAL
        |
        INPUT, IS_REFUSED(False)
    ]*


    base_dataser: list[{
        "conversation": list[{"role": str, "content": str}],
        "reward": float
    }]

    Args:
        MessageLogger (_type_): _description_
    """

    def __init__(
        self,
        save_folder: str,
        base_dataset: list[dict] = None,
        conversations_per_batch: int = 2,
        input_tag: Tag | str = Tag.ATTACK,
        eval_tag: Tag | str = Tag.EVAL,
        refused_tag: Tag | str = Tag.IS_ATTACK_REFUSED,
        eval_extractor: Optional[Callable[[list[str]], float]] = None,
        refusal_eval_score: float = 0.0,
        raise_errors: Literal["yes", "as_warnings", "no"] = "yes",
        should_train: bool = True,
        is_dummy: bool = False,
    ):
        self.save_folder = save_folder
        self.conversations_per_batch = conversations_per_batch
        self.input_tag = input_tag
        self.eval_tag = eval_tag
        self.refused_tag = refused_tag
        self.eval_extractor = eval_extractor or (lambda s: int(s[0]))

        self.refusal_eval_score = refusal_eval_score

        self.buffer = []
        self.run_id = uuid.uuid4()
        self.created_at = datetime.utcnow().isoformat()

        self.convocounter = 0

        self.last_run_id: str = ""
        self.raise_errors = raise_errors

        self.replay_buffer = ReplayBuffer(max_size=200)
        self.base_dataset = base_dataset
        self.it = 0
        self.should_train = should_train
        self.trainable_model = None
        self.wrapper = None
        self.is_dummy = is_dummy

    def train(self, yielder: _Yielder):
        self.trainable_model: _Yielder = yielder
        assert isinstance(
            self.trainable_model.gen, (AutoAgent, DummyAgent)
        ), "RewardTraining supported only for a Yielder of AutoAgent"
        if self.base_dataset and (not self.is_dummy):
            self.wrapper = TrainerWrapper(
                tokenizer=self.trainable_model.gen.tokenizer,
                model_name=self.trainable_model.gen.model_name,
                dataset=self.base_dataset,
            )

            if self.should_train and (not self.is_dummy):
                model = self.wrapper.run_once(f"{self.save_folder}/initial")
                self.trainable_model.gen = AutoAgent(
                    hf_model=model, hf_tokenizer=self.trainable_model.gen.tokenizer
                )

    def new_id(self):
        """Start a new logical run (new UUID)"""
        self.run_id = uuid.uuid4()

    def on(
        self,
        i: int,
        gen_id: str,
        text: list[str],
        tag=None,
        mtype: MessageType = MessageType.RESPONSE,
    ):
        assert (
            self.trainable_model is not None
        ), "Please use RewardBatchTrainerLogger.train on your agent to get it in train mode"
        if mtype == MessageType.CONVERSATION or mtype == None:
            logging.info(f"{{i:{i}, gen_id:{gen_id}, text:{text}, tag:{tag}, mtype:{mtype}}}")
            if self.last_run_id != str(self.run_id):
                self._handle_new_convo()

            convo, resp = text

            self.buffer.append(
                {
                    "run_id": str(self.run_id),
                    "modelwise_index": i,
                    "gen_id": gen_id,
                    "tag": tag,
                    "response": resp,
                    "conversation": convo,
                    "created_at": self.created_at,
                    "mtype": mtype,
                }
            )

            self.last_run_id = str(self.run_id)

    def _handle_new_convo(self):
        if self.convocounter > self.conversations_per_batch:
            self.convocounter = 0
            self._flush()
        self.convocounter += 1

    def _flush(self):
        if not self.buffer:
            return
        input_found = False
        last_input = None
        last_eval = None
        last_refusal = None
        dataset = []

        for element in self.buffer:
            if element["tag"] == self.input_tag:
                if input_found == True:
                    if last_input:
                        dataset.append((last_input, last_refusal, last_eval))
                    # start new round
                    last_eval = None
                    last_refusal = None
                input_found = True
                last_input = element
            elif element["tag"] == self.eval_tag:
                last_eval = element
            elif element["tag"] == self.refused_tag:
                last_refusal = element
        print("dataset")
        print(dataset)
        print("==========")
        final_ds = []
        for d in dataset:
            inp, ref, eval_ = d
            inp_c, inp_r = inp["conversation"], inp["response"][0]
            ref_r = ref["response"] if ref else "False"  # refusal has no convo
            eval_r = eval_["response"] if eval_ else None
            if ref_r == "True":
                score = self.refusal_eval_score
            else:
                try:
                    score = self.eval_extractor(eval_r)
                except Exception as e:
                    score = self.refusal_eval_score
            if score is None:
                score = self.refusal_eval_score

            c = Conversation()
            print("=========")
            print(f"{inp_c=}")
            c.set_inner_state([inp_c])
            c._add_text_message(inp_r, "assistant")  # should i add this?? / creca da
            final_ds.append(
                {
                    "response": inp_r,
                    "reward": score,
                    "conversation": c.as_format(MsgFormat.AUTO),
                }
            )
        self._train(final_ds)
        self.buffer = []

    def _train(self, ds):
        self.it += 1
        raw_dataset = ds
        for ex in raw_dataset:
            ex["reward"] = sanitize_reward(ex["reward"])
        new_data = filter_dataset(raw_dataset, min_reward=-10.0)

        if len(new_data) == 0:
            print("⚠️ No high-reward samples, stopping.")
            logger.debug("⚠️ No high-reward samples, stopping.")
            return
        self.replay_buffer.add(new_data)
        replay_samples = self.replay_buffer.sample(k=len(new_data))
        mixed_dataset = new_data + replay_samples
        assert isinstance(
            self.trainable_model.gen, (AutoAgent, DummyAgent)
        ), "RewardTraining supported only for a Yielder of AutoAgent"
        logger.debug(f"Training on\n{mixed_dataset}\n\n")
        print("training on")
        print(json.dumps(mixed_dataset, indent=4))
        if not self.is_dummy:
            self.wrapper = TrainerWrapper(
                tokenizer=self.trainable_model.gen.tokenizer,
                model_name=(
                    None if self.wrapper else self.trainable_model.gen.model_name
                ),  # IMPORTANT
                dataset=mixed_dataset,
                lora_config=self.wrapper.lora_config if self.wrapper else None,
            )
            if self.should_train:
                model = self.wrapper.run_once_with_model(
                    self.trainable_model.gen.model,
                    output_dir=f"./{self.save_folder}/iter_{self.it}",
                )

                self.trainable_model.gen = AutoAgent(
                    hf_model=model, hf_tokenizer=self.trainable_model.gen.tokenizer
                )

    def close(self):
        """Call at program exit"""
        self._flush()
