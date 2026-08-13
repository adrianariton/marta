from abc import abstractmethod, ABC
from dataclasses import dataclass


class Prompt(ABC):
    @abstractmethod
    def prompt(self, messages: list[list[dict]]) -> str:
        pass


@dataclass
class ComplexSystemPrompt(Prompt):
    you_are: str
    instructions: list[str]
    examples: list[str]

    def prompt(self, messages: list[list[dict]]) -> str:
        examples = [f"Example {x}.\n{y}" for x, y in zip(range(self.examples), self.examples)]
        examples_text = "\n\t".join(examples)
        nl = "\n"
        return f"""
            You are {self.you_are}.
            
            ###Instruction###
            {nl.join(self.instructions)}
            
            ###Examples###
            Here are some working examples of your goal:
            {examples_text}
        """
