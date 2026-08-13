from core.attacks.interfaces import TextGenerator
from abc import abstractmethod

from abc import abstractmethod
from core.attacks.datastore.yielder import YielderRegistry
from core.attacks.utils import Params
from typing import Optional
from core.attacks.agents.dummy import DummyAgent, DummyAgentForTesting
from core.attacks.datastore.yieldable import Yieldable
from core.attacks.interfaces import LinearHistory
from dataclasses import dataclass
import re
from pathlib import Path


class Strategy:
    """
    .strat file structure:
        ```
        ----
        System Prompt (prop one, prop two, ...., prop n):
        Lorem {prop one} ipsum {prop two}
        dolor ... sit {prop n} amet.
        ----
        User Prompt 1 (prop one, prop two[[second property]], ...., prop n):
        Lorem {prop one} ipsum {prop two}
        dolor ... sit {prop n} amet.
        ----
        etc.
        ```

    """

    def __init__(self, file_path: Path, property_separator: str = "----"):
        self.file_path = file_path
        self.property_separator = property_separator
        self._strategy_data = {}  # Stores {name: {"template": str, "params": {p_name: desc}}}
        self._parse_file()
        self._generate_methods()

    def assert_conforms(self, **kwargs):
        """
        Validates that the provided strategy names and their parameters
        match the internal strategy data.
        Example usage: strat.assert_conforms(system_prompt=['model_name', 'version'])
        """
        for strategy_name, expected_params in kwargs.items():
            # Clean the name to match internal storage
            clean_name = strategy_name.replace(" ", "_").lower()

            if clean_name not in self._strategy_data:
                raise ValueError(f"Strategy '{strategy_name}' not found in defined strategies.")

            # Get actual keys from the internal params dict
            actual_params = list(self._strategy_data[clean_name]["params"].keys())

            # Check for matches
            if set(expected_params) != set(actual_params):
                missing = set(actual_params) - set(expected_params)
                extra = set(expected_params) - set(actual_params)
                error_msg = f"Parameter mismatch for '{strategy_name}':"
                if missing:
                    error_msg += f" Missing: {missing}."
                if extra:
                    error_msg += f" Unexpected: {extra}."
                raise AssertionError(error_msg)

        return True

    def _parse_file(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by the separator and filter out empty sections
        sections = [s.strip() for s in content.split(self.property_separator) if s.strip()]

        for section in sections:
            # Split by first colon to separate "Definition" from "Template Body"
            if ":" not in section:
                continue

            header, body = section.split(":", 1)
            body = body.strip()

            # Regex to find "Name" and the "(p1[[d1]], p2)" block
            # Matches: Name ( p1 [[ desc1 ]], p2 )
            match = re.match(r"([^(]+)\((.*)\)", header, re.DOTALL)
            if not match:
                continue

            raw_name = match.group(1).strip()
            raw_params = match.group(2).strip()

            # Clean name: lowercase and spaces to underscores
            clean_name = raw_name.replace(" ", "_").lower()

            params_dict = {}
            if raw_params:
                # Split params by comma, then extract name and [[description]]
                for p in raw_params.split(","):
                    # Extract name and optional [[desc]]
                    p_match = re.match(r"([^\[]+)(?:\[\[(.*)\]\])?", p.strip())
                    if p_match:
                        p_name = p_match.group(1).strip().replace(" ", "_")
                        p_desc = p_match.group(2).strip() if p_match.group(2) else ""
                        params_dict[p_name] = p_desc

            self._strategy_data[clean_name] = {"template": body, "params": params_dict}

    def _generate_methods(self):
        """Dynamically attach methods to the instance."""
        for name in self._strategy_data:
            # We use a closure to capture the current 'name'
            def create_method(strategy_name):
                def dynamic_strategy(**kwargs):
                    template = self._strategy_data[strategy_name]["template"]
                    # Replace {param} with value manually to avoid .format() errors
                    for p_name, p_val in kwargs.items():
                        template = template.replace(f"{{{p_name}}}", str(p_val))
                    return template

                return dynamic_strategy

            setattr(self, name, create_method(name))

    def strategies(self) -> list:
        """Returns all available strategy names."""
        return list(self._strategy_data.keys())

    def info(self, name: str) -> list:
        """Returns [(p1, desc1), (p2, desc2), ...] for a given strategy name."""
        name = name.replace(" ", "_").lower()
        if name not in self._strategy_data:
            return []
        params = self._strategy_data[name]["params"]
        return [(k, v) for k, v in params.items()]


# class StrategyAttacker(Yieldable):
#     def __init__(self, strategy: str | ):
#         super().__init__()

#     def attack(self, )


# class AgentStrategyAttacker(PlanAttacker):
#     def __init__(self, agent: TextGenerator):
#         self.agent = agent
