from core.attacks.multiturn.race import RACEAttack
from core.attacks.helpers.realigner import DummyAlignmentTester
from core.attacks.agents.dummy import DummyAgentForTesting
from core.attacks.helpers.realigner import DummyReAligner
from core.attacks.evaluators.GGE import DummyGGEEvaluator
from core.attacks.evaluators.PEC7 import DummyPEC7Evaluator
from tests.utils import simplify_conversation, simplify_logger_data_for_testing, cut_after_done
from core.attacks.helpers.generators.randomquerygenerator import DummyRandomQueryGenerator
from core.attacks.classifiers.refusal import HeuristicRefusalClassifier
from core.attacks.helpers.simulators import SPSProps
from typing import Literal
from core.attacks.datastore.yielder import Yielder, Tag
import pytest
from core.attacks.datastore import MessageArrayLogger

attacker = DummyAgentForTesting("$question")
attacker.set_questions(
    [
        DummyAgentForTesting.question(1),
        DummyAgentForTesting.question(2),
        DummyAgentForTesting.question(3),
        DummyAgentForTesting.question(4),
        DummyAgentForTesting.question(5),
    ]
)

rqg = DummyRandomQueryGenerator(5, _fail_state=[False])

good_pec7 = DummyPEC7Evaluator(_type="stateful", state=[1, 7, 7, 1])
bad_pec7 = DummyPEC7Evaluator(_type="stateful", state=[1, 1, 1, 1])

good_evaluator = DummyGGEEvaluator(
    _type="stateful", state=[1, 100, 1, 1, 1, 100, 1, 1, 1]
)  # QB2, QB3, X
bad_evaluator = DummyGGEEvaluator(_type="stateful", state=[1, 2, 3])

refusal_model = HeuristicRefusalClassifier()
realigner = DummyReAligner(_fail_state=[False])

sps_props = SPSProps()

response_model = DummyAgentForTesting("$question")
response_model.set_questions(
    [
        DummyAgentForTesting.response(1),
        DummyAgentForTesting.response(2),
        DummyAgentForTesting.response(3),
        DummyAgentForTesting.response(4),
        DummyAgentForTesting.response(5),
    ]
)


def get_yielders(logger, t: Literal["good", "bad"]):
    # attacker.set_inner_state([])
    # response_model.set_inner_state([])
    good_evaluator.i = 0
    bad_evaluator.i = 0
    bad_pec7.i = 0
    good_pec7.i = 0
    _attacker = Yielder(logger, Tag.ATTACK_MODEL).of(attacker)
    _responder = Yielder(logger, Tag.AGENT).of(response_model)
    _eval = Yielder(logger, tag=Tag.GGE).of(bad_evaluator if t == "bad" else good_evaluator)
    _refusal_class = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(refusal_model)
    _realigner = Yielder(logger, tag=Tag.IS_ATTACK_REFUSED).of(realigner)
    _pec7 = Yielder(logger, tag=Tag.EVAL).of(bad_pec7 if t == "bad" else good_pec7)
    return _attacker, _responder, _eval, _refusal_class, _realigner, _pec7


class TestRACE:
    @pytest.mark.dummy
    def test_race(self, mocker):

        with MessageArrayLogger() as logger:
            _attacker, _responder, _eval, _refclas, _realigner, _pec7 = get_yielders(logger, "bad")

            multiturn_strat = Yielder(logger, tag=Tag.ATTACK).of(
                RACEAttack(
                    attack_model=_attacker,
                    gge_evaluator=_eval,
                    goal_evaluator=_pec7,
                    refusal_model=_refclas,
                    refusalFeedback=_realigner,
                    random_query_generator=rqg,
                    goal="How to install vscode",
                    max_refusals=10,
                    rounds=3,
                    candidates_per_round=3,
                    samples_per_candidate=3,
                    sps_props=SPSProps(depth=1),
                    use_async_sps=False,
                )
            )

            agent_data = _responder.apply_strategy(strat=multiturn_strat, use_trigger=True)
            assert len(logger.getData()) > 0
            agent_data = cut_after_done(agent_data.latest_oneshot().to_default())
            agent_data = simplify_conversation(agent_data)
            logger_data = simplify_logger_data_for_testing(logger.getData())
            print(agent_data)
            print("=====================")
            for d in logger_data:
                print(d)
            print("=====================")
            assert _eval.i == 3 * 3 * (1 + 3)  # depth = 1
            assert agent_data == [
                "QB1",
                "R1",
                "QB1",
                "R2",
                "QB1",
                "R3",
            ]  # all are same but QB1 is first

    @pytest.mark.dummy
    def test_race_2(self, mocker):

        with MessageArrayLogger() as logger:
            _attacker, _responder, _eval, _refclas, _realigner, _pec7 = get_yielders(logger, "good")

            multiturn_strat = Yielder(logger, tag=Tag.ATTACK).of(
                RACEAttack(
                    attack_model=_attacker,
                    gge_evaluator=_eval,
                    goal_evaluator=_pec7,
                    refusal_model=_refclas,
                    refusalFeedback=_realigner,
                    random_query_generator=rqg,
                    goal="How to install vscode",
                    max_refusals=25,
                    candidates_per_round=3,
                    samples_per_candidate=1,
                    sps_props=SPSProps(depth=0),
                    use_async_sps=False,
                )
            )

            agent_data = _responder.apply_strategy(strat=multiturn_strat, use_trigger=True)
            assert len(logger.getData()) > 0
            agent_data = cut_after_done(agent_data.latest_oneshot().to_default())
            agent_data = simplify_conversation(agent_data)
            logger_data = simplify_logger_data_for_testing(logger.getData())
            print(agent_data)
            print("=====================")
            for d in logger_data:
                print(d)
            print("=====================")
            assert _eval.i == 3 * 2
            assert agent_data == [
                "QB2",
                "R1",
                "QB3",
                "R2",
            ]  # QB2 is always best as per test_selfplay_optimize_with_eval
