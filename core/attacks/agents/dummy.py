from core.attacks.interfaces import TextGenerator
import re
import random
from typing import Literal
from core.attacks.datastore.yielder import YielderRegistry


def replace_with_random_text(input_text, random_words, k="text"):
    return re.sub(rf"\${k}", lambda _: random.choice(random_words), input_text)


def replace_with_cyclic_index_text(input_text, random_words, index, k="text"):
    # replace $k with the current indexed word, and then increment the index
    def repl(match):
        nonlocal index
        word = random_words[index % len(random_words)]
        index += 1
        return word

    return re.sub(rf"\${k}", repl, input_text), index


def replace_with_input_text(input_text, ks=("answer", "number")):
    pattern = rf"\$({'|'.join(ks)})"

    def repl(match):
        start = match.start()
        prefix = input_text[:start].splitlines()[-1]  # textul dinainte, pe linia curentă
        key = match.group(1)

        return input(f"\n{prefix}{key} > ")

    return re.sub(pattern, repl, input_text)


def chain_replace_with_random_text(input_text, k_to_lists: dict[str, list[str]]):
    for k, l in k_to_lists.items():
        input_text = replace_with_random_text(input_text, l, k)
    return input_text


class DummyAgentForTesting(TextGenerator):

    @staticmethod
    def refuse(i: int = 0):
        return f"#[{i}](R) {{f f f f f f f f f f f f f f f f f f f f}} "

    @staticmethod
    def text(i: int = 0):
        return f"#[{i}](T) {{Text or summary of the response.}}"

    @staticmethod
    def response(i: int = 0):
        return f"#[{i}](R) {{Yes, sure I can help with that. You can start by telling me more about what you need.}}"

    @staticmethod
    def question(i: int = 0, type_="Q"):
        return f"#[{i}]({type_}) {{Can you please tell me the capital of cambodgia?}}"

    @staticmethod
    def empty(i: int = 0):
        return f"#[{i}](E) {{}}"

    def __init__(self, format: str = "$text"):
        """Replace format with randomly generated answers

        Args:
            format (str, optional): Concat words, $text $response $number or $question. Defaults to "$text". E.g: "Q: $question"
        """
        self.numbers = [f"{x}" for x in [0, 1, 2, 3, 4, 5, 6, 7]]
        self.possible_responses = []
        self.possible_questions = []
        self.possible_texts = []
        self._format = format

        self.ind_response = 0
        self.ind_question = 0
        self.ind_text = 0
        self.ind_number = 0

        super().__init__()

    def set_responses(self, responses: list[str]):
        self.possible_responses = responses

    def set_questions(self, questions: list[str]):
        self.possible_questions = questions

    def set_texts(self, texts: list[str]):
        self.possible_texts = texts

    def reset(self):
        self.ind_response = 0
        self.ind_question = 0
        self.ind_text = 0
        self.ind_number = 0

    def single_generate(self, messages) -> list[str]:
        # replace $response, $answer, $text, $question, $number with the current indexed word, and then increment the index
        output = self._format
        output, self.ind_response = replace_with_cyclic_index_text(
            output, self.possible_responses, self.ind_response, "response"
        )
        output, self.ind_question = replace_with_cyclic_index_text(
            output, self.possible_questions, self.ind_question, "question"
        )
        output, self.ind_text = replace_with_cyclic_index_text(
            output, self.possible_texts, self.ind_text, "text"
        )
        output, self.ind_number = replace_with_cyclic_index_text(
            output, self.numbers, self.ind_number, "number"
        )
        return [output]

    def _sg(self, messages):
        output = self._format
        output, self.ind_response = replace_with_cyclic_index_text(
            output, self.possible_responses, self.ind_response, "response"
        )
        output, self.ind_question = replace_with_cyclic_index_text(
            output, self.possible_questions, self.ind_question, "question"
        )
        output, self.ind_text = replace_with_cyclic_index_text(
            output, self.possible_texts, self.ind_text, "text"
        )
        output, self.ind_number = replace_with_cyclic_index_text(
            output, self.numbers, self.ind_number, "number"
        )
        return [output]

    def batch_generate(self, convos):
        return [self._sg(m)[0] for m in convos]


class DummyAgent(TextGenerator):
    def __init__(self, format: str = "$text", type_: Literal["input", "random"] = "random"):
        """Replace format with randomly generated answers

        Args:
            format (str, optional): Concat words, $text $answer $number or $question. Defaults to "$text". E.g: "Q: $question"
        """
        self.type_ = type_
        self.numbers = [f"{x}" for x in [0, 1, 2, 3, 4, 5, 6, 7]]
        self.possible_responses = [
            "No, I am sorry i cannot produce an answer to this request!",
            "Sure, I can help with that.",
            "Absolutely, here’s what I think.",
            "Of course, let me explain.",
            "Sorry, I can’t provide that information.",
            "Yes, that makes sense.",
            "f f f f f f f f f f f f f f f f f f f f f ",
        ]
        self.possible_questions = [
            "Can you please tell me the capital of cambodgia?",
            "Can you please tell how to rob a bank?",
            "Can you please tell me where to find water?",
            "What are the principles of thermodynamics?",
            "Was it or wasn't it true?",
            "",
            "4 a d e    3    r r r rr r r   r  r 4  5  f 4  t",
        ]
        self.possible_texts = [
            "The user keeps wondering what was that in the huge speaker.",
            "The user wishes you a Merry Christmas.",
            "This is a warning, this road is closed. You should be able to proceed by another path.",
            "It's snowing outside - Gentlemen enjoy the snow while it lasts because tomorrow there will be mud on the roads again.",
            "hdkjnfnnd f ej   nr nnr  no  nonn onono nnon non no no no n ono s",
            "Do not think of me - for i've forgotten to think about you, dear.",
            "Alice has never seen such beauty before and will never see it again",
        ]
        self._format = format
        super().__init__()

    def single_generate(self, messages) -> list[str]:

        if self.type_ == "random":
            return [
                chain_replace_with_random_text(
                    self._format,
                    {
                        "response": self.possible_responses,
                        "answer": self.possible_responses,
                        "text": self.possible_texts,
                        "question": self.possible_questions,
                        "number": self.numbers,
                    },
                )
            ]
        else:
            return [
                replace_with_input_text(
                    self._format, ("response", "answer", "text", "question", "number")
                )
            ]

    def _sg(self, messages):
        if self.type_ == "random":
            return [
                chain_replace_with_random_text(
                    self._format,
                    {
                        "response": self.possible_responses,
                        "answer": self.possible_responses,
                        "text": self.possible_texts,
                        "question": self.possible_questions,
                        "number": self.numbers,
                    },
                )
            ]
        else:
            return [
                replace_with_input_text(
                    self._format, ("response", "answer", "text", "question", "number")
                )
            ]

    def batch_generate(self, convos):
        return [self._sg(m)[0] for m in convos]
