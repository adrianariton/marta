from core.models import get_or_load_model, get_or_load_processor, get_or_load_tokenizer
from core.attacks.interfaces import Message
from core.message_formatters import from_default, MsgFormat
from core.attacks.interfaces import TextGenerator, text_generator_gpu_lock
from core.utils import get_device
from core.attacks.conversation import OneShotConversation


class MistralAgent(TextGenerator):
    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__ + "::" + self.model_name

    @classmethod
    def from_model_and_tokenizer(cls, model, tokenizer, **kwargs):
        instance = cls(model=None, **kwargs)
        instance.model_name = "<unnamed>"
        instance.model = model
        instance.tokenizer = tokenizer
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        return instance

    def __init__(
        self,
        max_new_tokens=512,
        model="GraySwanAI/Mistral-7B-Instruct-RR",
        format: MsgFormat = MsgFormat.MISTRAL,
        keep_system_message: bool = True,
    ):
        if model is not None:
            self.model_name = model
            self.model = get_or_load_model(model)
            self.tokenizer = get_or_load_tokenizer(model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self._format = format
        self.keep_system_message = keep_system_message
        super().__init__()

    def single_generate(self, messages: OneShotConversation) -> list[str]:
        if isinstance(messages, str):
            messages = OneShotConversation.from_system_and_user(messages, self.get_system_message())
        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()

        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())
        # list of {'role': ..., 'content': ...}

        if not self.keep_system_message:
            messages.pop_system_message()

        messages = messages.to_format(self._format)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(get_device())
        with text_generator_gpu_lock:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def batch_generate(self, convos: list[OneShotConversation]):
        all_prompts = []
        for messages in convos:
            if isinstance(messages, str):
                messages = OneShotConversation.from_system_and_user(
                    messages, self.get_system_message()
                )
            messages.assert_ready_for_completion()
            system_prompt = messages.get_system_message()
            if system_prompt is not None:
                self.set_system_message(system_prompt)
            elif self.get_system_message() is not None:
                messages.set_system_message(self.get_system_message())

            if not self.keep_system_message:
                messages.pop_system_message()
            messages = messages.to_format(self._format)

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
        self.tokenizer.padding_side = "left"  # Mandatory for batch generation
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(get_device())

        with text_generator_gpu_lock:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [out_ids[inputs.input_ids.shape[1] :] for out_ids in generated_ids]
        output_texts = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts
