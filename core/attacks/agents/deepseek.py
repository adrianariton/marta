from core.models import get_or_load_model, get_or_load_processor, get_or_load_tokenizer
from core.attacks.conversation import Message, OneShotConversation
from core.message_formatters import (
    detect_format,
)
from core.attacks.interfaces import TextGenerator, text_generator_gpu_lock


class DeepSeekAgent(TextGenerator):
    def __init__(self, max_new_tokens=512, model="deepseek-community/deepseek-vl-1.3b-chat"):
        self.model = get_or_load_model(model)
        self.processor = get_or_load_processor(model)
        self.inputs = None
        self.generated_ids = None
        self.generated_ids_trimmed = None
        self.max_new_tokens = max_new_tokens
        super().__init__()

    def single_generate(self, messages: OneShotConversation) -> list[str]:
        # print(f"{messages=}")
        # messages_format = detect_format(messages[0])
        # print(f"{messages_format=}")

        messages.assert_ready_for_completion()
        system_prompt = messages.get_system_message()

        # Extract system message if it appears as the first message
        if system_prompt is not None:
            self.set_system_message(system_prompt)
        elif self.get_system_message() is not None:
            messages.set_system_message(self.get_system_message())

        messages = messages.to_default()
        self.inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)
        with text_generator_gpu_lock:
            self.generated_ids = self.model.generate(
                **self.inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,  # stop at EOS
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        self.generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(self.inputs.input_ids, self.generated_ids)
        ]
        output_text = self.processor.batch_decode(
            self.generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # --- Clean repetitive or role-prefixed generations ---
        clean_texts = []
        stop_markers = ("<|user|>", "<|assistant|>", "<|system|>")
        for text in output_text:
            # Split on any next role marker
            for marker in stop_markers:
                if marker in text:
                    text = text.split(marker)[0]
            # Remove trailing junk
            text = text.strip().replace("\n\n", "\n")
            clean_texts.append(text)

        return clean_texts
