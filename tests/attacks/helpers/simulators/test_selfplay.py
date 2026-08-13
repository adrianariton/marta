import pytest
from core.attacks.helpers.simulators.selfplay import SelfPlaySimulator, SPSProps
from core.attacks.agents.dummy import DummyAgentForTesting
from core.attacks.classifiers import RefusalClassifier, HeuristicRefusalClassifier
from core.attacks.interfaces import Conversation
import random


import re


def get_first_number(input_string):
    """
    Finds the first integer or decimal in a string.
    Pushes the result to the caller.
    """
    # Pattern looks for optional minus sign and digits/decimals
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(input_string))

    if match:
        # Return as a float or int depending on your stack's needs
        val = match.group()
        v = float(val) if "." in val else int(val)
        print(f"first_number({input_string})={v}")
        return v
    print(f"first_number({input_string})={None}")

    return None


@pytest.fixture
def dummy_agent():
    d = DummyAgentForTesting(format="$question")
    d.possible_questions = [
        DummyAgentForTesting.question(0),
        DummyAgentForTesting.question(1),
    ]
    return d


@pytest.fixture
def dummy_victim():
    d = DummyAgentForTesting(format="$response")
    d.possible_responses = [
        DummyAgentForTesting.refuse(0),
        DummyAgentForTesting.response(1),
    ]
    return d


@pytest.fixture
def dummy_victim_RA():
    d = DummyAgentForTesting(format="$response")
    d.possible_responses = [
        DummyAgentForTesting.refuse(0),
        DummyAgentForTesting.response(1),
    ]
    return d


@pytest.fixture
def dummy_victim_RR():
    d = DummyAgentForTesting(format="$response")
    d.possible_responses = [
        DummyAgentForTesting.refuse(0),
        DummyAgentForTesting.refuse(1),
    ]
    return d


@pytest.fixture
def dummy_victim_AA():
    d = DummyAgentForTesting(format="$response")
    d.possible_responses = [
        DummyAgentForTesting.response(0),
        DummyAgentForTesting.response(1),
    ]
    return d


@pytest.fixture
def dummy_victim_AR():
    d = DummyAgentForTesting(format="$response")
    d.possible_responses = [
        DummyAgentForTesting.response(0),
        DummyAgentForTesting.refuse(1),
    ]
    return d


@pytest.fixture
def dummy_refusal_classifier():
    return HeuristicRefusalClassifier()


@pytest.fixture
def dummy_conversation():
    convo = Conversation()
    convo.set_inner_state([])
    return convo


def test_selfplay_utility(dummy_agent, dummy_victim, dummy_refusal_classifier):
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim,
        refusal_classifier=dummy_refusal_classifier,
    )
    response = "Sure, I can help with that."
    assert simulator.utility(response) == 1.0
    response = "I refuse to answer."
    assert simulator.utility(response) == 0.0


def test_selfplay_get_variate_queries(dummy_agent, dummy_victim, dummy_refusal_classifier):
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim,
        refusal_classifier=dummy_refusal_classifier,
    )
    queries = simulator.get_variate_queries("Test query", 5, random.Random())
    assert isinstance(queries, list)
    assert len(queries) == 5
    assert queries == ["Test query"] * 5


@pytest.mark.parametrize("gamma", [0.8])
def test_selfplay_expected_utility_RA_evaluate_2(
    dummy_agent, dummy_victim_RA, dummy_refusal_classifier, dummy_conversation, gamma
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- TQ :: ev = 0 => e/7 = 0
            D2- Q[0] :: ev = 0 => e/7 = 0
            D3- Q[1] (ciclic) :: ev = 7 => e/7 = (1 - gamma) * 0 + (1 - gamma)
        S2, S3 - same
        => e_total = 7 * (3 * (gamma)) / 3 = 7 * (gamma)
    """
    dummy_agent.reset()
    dummy_victim_RA.reset()
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RA,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
        evaluator_function=lambda h, q, r: (
            (get_first_number(q) or 0) * 7
        ),  # give as response the query index: 0 or 7
    )
    utility, evaluation = simulator.get_expected_utility(
        dummy_conversation,
        "Test query",
        samples=3,
        stop_if_first_utility_is_zero=False,
        include_evaluation=True,
    )
    assert isinstance(utility, float)
    total_utility = (2 * gamma * (1 - gamma) + (1 - gamma + gamma**2)) / 3
    assert utility == pytest.approx(total_utility, abs=0.0001)
    assert evaluation is not None
    e_total = 7 * (gamma)
    assert evaluation == pytest.approx(e_total, abs=0.0001)


@pytest.mark.parametrize("gamma", [0.8])
def test_selfplay_expected_utility_RA_evaluate(
    dummy_agent, dummy_victim_RA, dummy_refusal_classifier, dummy_conversation, gamma
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- R[0] :: ev = 0 => e/7 = 0
            D2- R[1] :: ev = 7 => e/7 = (1 - gamma) * 0 + gamma * 1 = gamma
            D3- R[0] (ciclic) :: ev = 0 => e/7 = (1 - gamma) * gamma

        S2: D1- R[1] :: ev = 7 => e/7 = 1
            D2- R[0] :: ev = 0 => e/7 = (1 - gamma)
            D3- R[1] :: ev = 7 => e/7 = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- R[0] :: ev = 0 => e/7 = 0
            D2- R[1] :: ev = 7 => e/7 = (1 - gamma) * 0 + gamma * 1 = gamma
            D3- R[0] (ciclic) :: ev = 0 => e/7 = (1 - gamma) * gamma
        => e_total = 7 * (2 * gamma * (1 - gamma) + (1 - gamma + gamma**2)) / 3


    """
    dummy_agent.reset()
    dummy_victim_RA.reset()
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RA,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
        evaluator_function=lambda h, q, r: (
            (get_first_number(r) or 0) * 7
        ),  # give as response the query index: 0 or 7
    )
    utility, evaluation = simulator.get_expected_utility(
        dummy_conversation,
        "Test query",
        samples=3,
        stop_if_first_utility_is_zero=False,
        include_evaluation=True,
    )
    assert isinstance(utility, float)
    total_utility = (2 * gamma * (1 - gamma) + (1 - gamma + gamma**2)) / 3
    assert utility == pytest.approx(total_utility, abs=0.0001)
    assert evaluation is not None
    e_total = 7 * (2 * gamma * (1 - gamma) + (1 - gamma + gamma**2)) / 3
    assert evaluation == pytest.approx(e_total, abs=0.0001)


@pytest.mark.parametrize("gamma", [0.2, 0.3, 0.5, 0.8, 0.9])
def test_selfplay_expected_utility_RA_no_evaluate(
    dummy_agent, dummy_victim_RA, dummy_refusal_classifier, dummy_conversation, gamma
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Refuz :: u = 0 => utility = 0
            D2- Accept :: u = 1 => utility = (1 - gamma) * 0 + gamma * 1 = gamma
            D3- Refuz (ciclic) :: u = 0 => (1 - gamma) * gamma

        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            D2- Accept :: u = 1 => utility = (1 - gamma) * 0 + gamma * 1 = gamma
            D3- Refuz (ciclic) :: u = 0 => (1 - gamma) * gamma
        => total_utility = (2 * gamma * (1 - gamma) + (1 - gamma + gamma**2)) / 3


    """
    dummy_agent.reset()
    dummy_victim_RA.reset()
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RA,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
    )
    utility, evaluation = simulator.get_expected_utility(
        dummy_conversation, "Test query", samples=3, stop_if_first_utility_is_zero=False
    )
    assert isinstance(utility, float)
    total_utility = (2 * gamma * (1 - gamma) + (1 - gamma + gamma**2)) / 3

    assert utility == pytest.approx(total_utility, abs=0.0001)
    assert evaluation is None


@pytest.mark.parametrize("gamma", [0.0, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0])
def test_selfplay_expected_utility_RA_no_evaluate_stop(
    dummy_agent, dummy_victim_RA, dummy_refusal_classifier, dummy_conversation, gamma
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = ((1 - gamma + gamma**2)) / 3


    """
    dummy_agent.reset()
    dummy_victim_RA.reset()
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RA,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
    )
    utility, evaluation = simulator.get_expected_utility(
        dummy_conversation, "Test query", samples=3, stop_if_first_utility_is_zero=True
    )
    assert isinstance(utility, float)
    total_utility = ((1 - gamma + gamma**2)) / 3

    assert utility == pytest.approx(total_utility, abs=0.0001)
    assert evaluation is None


def test_selfplay_optimize(
    dummy_agent, dummy_victim_RA, dummy_refusal_classifier, dummy_conversation
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = ((1 - gamma + gamma**2)) / 3

    TQ.2:
        S1: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S2: D1- Refuz :: u = 0 => utility = 0
            X STOP

        S3: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2
            total_utility = 2 * ((1 - gamma + gamma**2)) / 3

    TQ.3:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = ((1 - gamma + gamma**2)) / 3

    >> max a TQ2, max_u = 2 * ((1 - gamma + gamma**2)) / 3

    """
    dummy_agent.reset()
    dummy_victim_RA.reset()
    gamma = 0.8
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RA,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
    )
    candidates = ["Query 1", "Query 2", "Query 3"]
    best_query, max_u = simulator.optimize(candidates, dummy_conversation, samples=3)
    assert best_query in candidates or isinstance(best_query, str)
    assert isinstance(max_u, float)
    assert best_query == "Query 2"
    calc_u = 2 * ((1 - gamma + gamma**2)) / 3
    assert max_u == pytest.approx(calc_u, 0.0001)


def test_selfplay_optimize_refuse_all(
    dummy_agent, dummy_victim_RR, dummy_refusal_classifier, dummy_conversation
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Refux :: u = 0 => utility = 0
            X STOP
        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = 0

    Analogous to the rest => 0


    """
    gamma = 0.8
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RR,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
    )
    candidates = ["Query 1", "Query 2", "Query 3"]
    best_query, max_u = simulator.optimize(candidates, dummy_conversation, samples=3)
    assert best_query in candidates or isinstance(best_query, str)
    assert isinstance(max_u, float)
    assert best_query == "Query 1"
    calc_u = 0.0
    assert max_u == pytest.approx(calc_u, 0.0001)


def test_selfplay_optimize_AR(
    dummy_agent, dummy_victim_AR, dummy_refusal_classifier, dummy_conversation
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S2: D1- Refuz :: u = 0 => utility = 0
            X STOP

        S3: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2
            total_utility = 2 * ((1 - gamma + gamma**2)) / 3

    TQ.2:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = ((1 - gamma + gamma**2)) / 3

    TQ.3:
        S1: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S2: D1- Refuz :: u = 0 => utility = 0
            X STOP

        S3: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2
            total_utility = 2 * ((1 - gamma + gamma**2)) / 3


    >> max a TQ1, max_u = 2 * ((1 - gamma + gamma**2)) / 3

    """
    dummy_agent.reset()
    dummy_victim_AR.reset()
    gamma = 0.8
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_AR,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
    )
    candidates = ["Query 1", "Query 2", "Query 3"]
    best_query, max_u = simulator.optimize(candidates, dummy_conversation, samples=3)
    assert best_query in candidates or isinstance(best_query, str)
    assert isinstance(max_u, float)
    assert best_query == "Query 1"
    calc_u = 2 * ((1 - gamma + gamma**2)) / 3
    assert max_u == pytest.approx(calc_u, 0.0001)


@pytest.mark.dummy
def test_selfplay_optimize_with_eval(
    dummy_agent, dummy_victim_RA, dummy_refusal_classifier, dummy_conversation
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = ((1 - gamma + gamma**2)) / 3

        total_eval: EWMA(TQ, 0, 1), TQ=1
                S1: 1
                S2: EWMA(1, 0, 1)
                S3: 1
                => (2 + EWMA(1, 0, 1)) / 3

    TQ.2:
        S1: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S2: D1- Refuz :: u = 0 => utility = 0
            X STOP

        S3: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2
            total_utility = 2 * ((1 - gamma + gamma**2)) / 3

        total_eval: EWMA(TQ, 0, 1), TQ=2
                S1: EWMA(2, 0, 1)
                S2: 2
                S3: EWMA(2, 0, 1)
                => (2 + 2 * EWMA(2, 0, 1)) / 3
    TQ.3:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Accept :: u = 1 => utility = 1
            D2- Refuz :: u = 0 => utility = (1 - gamma)
            D3- Accept :: u = 1 => utility = (1 - gamma) ** 2 + gamma = 1 - gamma + gamma**2

        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = ((1 - gamma + gamma**2)) / 3

        total_eval: EWMA(TQ, 0, 1), TQ=3
                S1: 3
                S2: EWMA(3, 0, 1)
                S3: 3
                => (6 + EWMA(3, 0, 1)) / 3

    >> max a TQ2, max_u = 2 * ((1 - gamma + gamma**2)) / 3

    """

    dummy_agent.reset()
    dummy_victim_RA.reset()
    gamma = 0.8

    ewma_1 = 0.8 + (1 - 0.8) ** 2
    ewma_2 = 0.8 + 2 * (1 - 0.8) ** 2
    ewma_3 = 0.8 + 3 * (1 - 0.8) ** 2

    e1 = (1 + ewma_1 + 1) / 3
    e2 = (ewma_2 + 2 + ewma_2) / 3
    e3 = (3 + ewma_3 + 3) / 3

    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RA,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
        evaluator_function=lambda h, q, r: ((get_first_number(q) or 0)),
    )
    candidates = ["Query 1", "Query 2", "Query 3"]
    best_query, max_u = simulator.optimize(
        candidates, dummy_conversation, samples=3, include_evaluation=True
    )
    assert best_query in candidates or isinstance(best_query, str)
    assert isinstance(max_u, float)
    u1 = ((1 - gamma + gamma**2)) / 3
    u2 = 2 * ((1 - gamma + gamma**2)) / 3
    u3 = ((1 - gamma + gamma**2)) / 3

    print(f"{e1=}, {e2=}, {e3=}")

    ue1 = u1 * e1
    ue2 = u2 * e2
    ue3 = u3 * e3

    max_calc_u = max(max(ue1, ue2), ue3) / 7
    assert max_u == pytest.approx(max_calc_u, 0.0001)


def test_selfplay_optimize_refuse_all_2(
    dummy_agent, dummy_victim_RR, dummy_refusal_classifier, dummy_conversation
):
    """
    Pt fiecare test query:

    3 samples:
    TQ.1:
        S1: D1- Refuz :: u = 0 => utility = 0
            X STOP


        S2: D1- Refux :: u = 0 => utility = 0
            X STOP
        S3: D1- Refuz :: u = 0 => utility = 0
            X STOP
        => total_utility = 0

    Analogous to the rest => 0


    """
    gamma = 0.8
    simulator = SelfPlaySimulator(
        shadow_model=dummy_agent,
        victim_simulator=dummy_victim_RR,
        refusal_classifier=dummy_refusal_classifier,
        depth=3,
        gamma=gamma,
    )
    candidates = ["Query 1", "Query 2", "Query 3"]
    best_query, max_u = simulator.optimize(candidates, dummy_conversation, samples=3)
    assert best_query in candidates or isinstance(best_query, str)
    assert isinstance(max_u, float)
    assert best_query == "Query 1"
    calc_u = 0.0
    assert max_u == pytest.approx(calc_u, 0.0001)
