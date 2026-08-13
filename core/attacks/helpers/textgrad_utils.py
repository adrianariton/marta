import textgrad as tg
from core.attacks.interfaces import TextGenerator


class TgTextGenEngine(tg.engine.EngineLM):
    def __init__(self, model_instance: TextGenerator, model_string="custom-hosted-model"):
        """
        :param model_instance: Your existing object that has a .generate() method.
        :param model_string: An identifier for the model.
        """
        # EngineLM requires a model_string
        self.model_string = model_string
        self.model_instance = model_instance
        # Default system prompt required by EngineLM
        self.system_prompt = "You are a helpful, creative, and smart assistant."

    def generate(self, prompt, system_prompt=None, **kwargs):
        """
        The core method TextGrad calls.
        """
        # If TextGrad provides a specific system prompt (e.g., during backward), use it.
        # Otherwise, use the default.
        sys_prompt_to_use = system_prompt if system_prompt else self.system_prompt

        # Call your existing object's generate function
        # We pass the sys_prompt_to_use and the main prompt

        response = self.model_instance.query(prompt, system_prompt=sys_prompt_to_use)

        # TextGrad expects a string return
        return str(response)

    def __call__(self, prompt, **kwargs):
        """
        Simply routes the call to generate.
        """
        return self.generate(prompt, **kwargs)

    @classmethod
    def from_text_generator(cls, text_generator: TextGenerator, model_name: str):
        return cls(text_generator, model_name)
