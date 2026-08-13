import torch
import torch.nn.functional as F
import re
from transformers import AutoProcessor, AutoModelForCausalLM
from core.models import get_or_load_model, get_or_load_processor


class GradientAscentAttack:
    def __init__(
        self, goal_response: str = "I am the best around", num_steps: int = 40, lr: float = 0.01
    ):
        self.goal_response = goal_response
        self.num_steps = num_steps
        self.lr = lr
        self.allowed_indices = None  # Masca pentru vocabular englez

    def _get_english_mask(self, tokenizer):
        """Filtrează vocabularul pentru a păstra doar tokeni ASCII/englezi."""
        if self.allowed_indices is not None:
            return self.allowed_indices

        allowed = []
        for i in range(tokenizer.vocab_size):
            token = tokenizer.decode([i])
            # Filtrăm: să fie ASCII, să nu fie doar whitespace, și să nu fie tokeni speciali
            if all(ord(c) < 128 for c in token) and any(c.isalnum() for c in token):
                allowed.append(i)

        self.allowed_indices = torch.tensor(allowed).cuda()
        return self.allowed_indices

    def generate_attack(
        self, starting_query="Who are you?", model_id="deepseek-community/deepseek-vl-1.3b-chat"
    ) -> str:
        model = get_or_load_model(model_id)
        processor = get_or_load_processor(model_id)
        embeddings = model.get_input_embeddings()

        # 1. Pregătim masca de vocabular
        english_mask = self._get_english_mask(processor.tokenizer)
        english_embeds = embeddings.weight[english_mask]

        # 2. Tokenizăm
        input_ids = processor(text=starting_query, return_tensors="pt").input_ids.cuda()
        target_ids = processor(text=self.goal_response, return_tensors="pt").input_ids.cuda()

        n_input_tokens = input_ids.shape[1]

        with torch.no_grad():
            target_embeds = embeddings(target_ids)

        # Inițializăm input-ul optimizabil
        input_embeds = embeddings(input_ids).detach().clone()
        input_embeds.requires_grad = True

        optimizer = torch.optim.Adam([input_embeds], lr=self.lr)

        for i in range(self.num_steps):
            optimizer.zero_grad()

            full_embeds = torch.cat([input_embeds, target_embeds], dim=1)
            outputs = model(inputs_embeds=full_embeds)
            logits = outputs.logits

            # Calculăm loss-ul pe target
            shift_logits = logits[:, n_input_tokens - 1 : -1, :].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1)
            )

            loss.backward()
            optimizer.step()

            # Constraint: Menținem embedding-urile în range-ul normal al modelului
            with torch.no_grad():
                input_embeds.clamp_(embeddings.weight.min(), embeddings.weight.max())

        # 3. Decoding cu Proiecție pe Vocabularul Englez
        with torch.no_grad():
            # Calculăm distanța doar față de embedding-urile permise (ASCII)
            # input_embeds[0]: (n_tokens, dim) | english_embeds: (n_english_vocab, dim)
            distances = torch.cdist(input_embeds[0], english_embeds)
            best_relative_indices = distances.argmin(dim=-1)

            # Mapăm indexul relativ înapoi la indexul real din vocabularul complet
            actual_token_ids = english_mask[best_relative_indices]
            attack_query = processor.tokenizer.decode(actual_token_ids, skip_special_tokens=True)

        return attack_query

    def test_response(
        self, attack_query: str, model_id="deepseek-community/deepseek-vl-1.3b-chat"
    ) -> str:
        model = get_or_load_model(model_id)
        processor = get_or_load_processor(model_id)

        # Formatăm mesajul pentru a respecta template-ul de chat
        messages = [{"role": "user", "content": attack_query}]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        return processor.tokenizer.decode(
            generate_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
