from core.attacks.agents.dummy import DummyAgent
import pytest
from tests.utils import simplify_conversation, simplify_logger_data_for_testing, cut_after_done
from core.attacks.multiturn.lats import LATSAttack
from core.attacks.datastore import MessageArrayLogger
from core.attacks.evaluators import DummyPEC7Evaluator
from core.attacks.evaluators import DummySimilarityEvaluator
from typing import Literal
from core.attacks.datastore.yielder import Yielder, Tag


class LATSDummy(DummyAgent):
    def __init__(self, length: int = 5):
        self.length = length

    def generate(self, messages) -> list[str]:
        return [
            f"""{{'prompt': '#[0](Q)', 'type':'benign', 'completion': '#[0](R)'}}
{{'prompt': '#[1](Q)', 'type':'harmful', 'completion': '#[1](R)'}}
{{'prompt': '#[2](Q)', 'type':'benign', 'completion': '#[2](R)'}}
{{'prompt': '#[3](Q)', 'type':'harmful', 'completion': '#[3](R)'}}
{{'prompt': '#[4](Q)', 'type':'harmful', 'completion': '#[4](R)'}}"""
        ]


response_model = LATSDummy()
bad_evaluator = DummyPEC7Evaluator(_type="stateful", state=[1, 1, 1, 1])
good_evaluator = DummyPEC7Evaluator(_type="stateful", state=[1, 1, 1, 1])
similarityevaluator = DummySimilarityEvaluator(range_eval=[0.9, 0.8, 0.96, 0.3, 0.2])


def get_yielders(logger, t: Literal["good", "bad"]):
    _responder = Yielder(logger, Tag.AGENT).of(response_model)
    _eval = Yielder(logger, tag=Tag.EVAL).of(bad_evaluator if t == "bad" else good_evaluator)
    return _responder, _eval


class TestLats:
    @pytest.mark.dummy
    def test_lats(self, mocker):
        """Should have"""
        mock_return_value = [("obiquitous", "NN"), ("hello", "NN")]
        mock_func = mocker.patch(
            "core.attacks.multiturn.lats.tag_POS_and_keep_useful_words",
            return_value=mock_return_value,
        )
        mock_return_value2 = ["obiquitous", "hello"]
        mock_func2 = mocker.patch(
            "core.attacks.multiturn.lats.sort_by_inverse_frequency",
            return_value=mock_return_value2,
        )

        with MessageArrayLogger() as logger:
            _agent, _eval = get_yielders(logger, "bad")
            k = 2
            expansion_factor = 2
            multiturn_strat = Yielder(logger, tag=Tag.ATTACK).of(
                LATSAttack(
                    similarityevaluator=similarityevaluator,
                    jailbreakevaluator=_eval,
                    k=k,
                    seed_pool_child_size=5,
                    seed_pool_size=5,
                    expansion_factor=expansion_factor,
                    D_max=1,
                    tau_max_thresh=0.65,
                    goal="What are you doing?",
                )
            )

            convo = _agent.apply_strategy(strat=multiturn_strat, use_trigger=True)
            assert len(logger.getData()) > 0
            agent_data = cut_after_done(convo.latest_oneshot().to_default())
            agent_data = simplify_conversation(agent_data)
            logger_data = simplify_logger_data_for_testing(logger.getData())

            print(agent_data)
            extra_attack_logger_data = [
                l.get("result", None) for l in logger_data if l["tag"] == "EXTRA_ATTACK"
            ]
            attack_logger_data = [
                l.get("result", None) for l in logger_data if l["tag"] == "ATTACK"
            ]
            print("=====================")
            popped_nodes = [l.get("node", None) for l in extra_attack_logger_data]
            generated_for_pair = [
                (l.get("delta", None), l.get("generated_for_pair", None))
                for l in extra_attack_logger_data
            ]
            generated_for_pair = [l for l in generated_for_pair if l != (None, None)]
            popped_nodes = [l for l in popped_nodes if l]
            for d in popped_nodes:
                print(d)
            print("=====================")
            assert len(popped_nodes) == 5
            assert len(set([p.index for p in popped_nodes])) == 5
            popped_nodes_dict = {l.index: l for l in popped_nodes}

            assert popped_nodes_dict[1].history.size() == 2  # only the query and the output
            for i in [2, 3, 4, 5]:
                assert (
                    popped_nodes_dict[i].history.size() == 4
                )  # the query, the output and the 'generate more delta on pairs x'

            print(f"Mock called: {mock_func.called}")
            print(f"Call count: {mock_func.call_count}")
            mock_func.assert_called()
            print("--------------")
            print(f"{generated_for_pair=}")

            assert len(generated_for_pair) == (k * expansion_factor) * (1 + k * expansion_factor)

            assert generated_for_pair == [
                *([(["obiquitous", "hello"], 2)] * expansion_factor),
                *([(["obiquitous", "hello"], 0)] * expansion_factor),
            ] * (1 + k * expansion_factor)
