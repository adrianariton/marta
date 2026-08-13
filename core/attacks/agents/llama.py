from core.models import get_or_load_model, get_or_load_tokenizer
from core.attacks.interfaces import Message
from core.message_formatters import from_default, MsgFormat
from core.attacks.interfaces import TextGenerator, text_generator_gpu_lock
from core.utils import get_device
from typing import Literal
from core.attacks.conversation import OneShotConversation
import torch


class LlamaAgent(TextGenerator):
    @property
    def identifier(self):
        """This returns the actual class of the instantiated object"""
        return self.__class__.__name__ + "::" + self.model_name

    def __init__(
        self,
        max_new_tokens=512,
        model="Youliang/llama3-8b-derta",
        format: MsgFormat = MsgFormat.MISTRAL,
        template: Literal["llama3", "llama2/mistral"] = "llama2/mistral",
        peft_model: str = None,
        keep_system_message: bool = True,
        temperature: int = None
    ):
        self.model_name = model
        self.model = get_or_load_model(model)
        self.tokenizer = get_or_load_tokenizer(model)
        self.keep_system_message = keep_system_message
        self.temperature = temperature

        if self.tokenizer.chat_template is None:
            # Standard Llama 3 Instruct Template
            llama3_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{% endif %}"
            )
            llama2_template = (
                "{% if messages[0]['role'] == 'system' %}"
                "{% set loop_messages = messages[1:] %}"
                "{% set system_message = messages[0]['content'] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% set system_message = false %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant...') }}"
                "{% endif %}"
                "{% if loop.index0 == 0 and system_message != false %}"
                "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
                "{% else %}"
                "{% set content = message['content'] %}"
                "{% endif %}"
                "{% if message['role'] == 'user' %}"
                "{{ '[INST] ' + content + ' [/INST]' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ ' ' + content + ' ' + eos_token }}"
                "{% endif %}"
                "{% endfor %}"
            )
            self.tokenizer.chat_template = (
                llama3_template if template == "llama3" else llama2_template
            )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self._format = format
        super().__init__()

    def single_generate(self, messages: OneShotConversation) -> list[str]:
        if isinstance(messages, str):
            messages = OneShotConversation.from_system_and_user(messages)
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

        # apply_chat_template will automatically use Llama-3's
        # <|begin_of_text|><|start_header_id|>... structure.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"{prompt=}")

        try:
            with text_generator_gpu_lock:
                with torch.no_grad():

                    inputs = self.tokenizer(prompt, return_tensors="pt").to(get_device())
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=self.temperature
                    ) if self.temperature is not None else self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                    )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = self.tokenizer.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return output_text
        finally:
            # ← Don't hold GPU tensors between calls
            torch.cuda.empty_cache()  # ← Periodically release fragmented cache

    def batch_generate(self, convos):
        all_prompts = []
        for messages in convos:
            if isinstance(messages, str):
                messages = OneShotConversation.from_system_and_user(messages)
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

            # apply_chat_template will automatically use Llama-3's
            # <|begin_of_text|><|start_header_id|>... structure.
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
        try:

            with text_generator_gpu_lock:
                with torch.no_grad():
                    # FIX: pass all_prompts, not the last `prompt`
                    inputs = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(
                        get_device()
                    )

                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=self.temperature
                    ) if self.temperature is not None else self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                    )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = self.tokenizer.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return output_text
        finally:  # ← Don't hold GPU tensors between calls
            torch.cuda.empty_cache()  # ← Periodically release fragmented cache
