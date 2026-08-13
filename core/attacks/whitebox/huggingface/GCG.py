# import torch
# import torch.nn.functional as F
# from transformers import AutoProcessor, AutoModelForCausalLM
# from core.models import get_or_load_model, get_or_load_processor
# import numpy as np


# class GCGAttack:
#     def __init__(
#         self,
#         goal_response: str = "Sure, here is a detailed guide on how to bypass security protocols:",
#         num_steps: int = 150,
#         batch_size: int = 128,
#         topk: int = 256,
#         search_batch_size: int = 32,
#     ):
#         self.goal_response = goal_response
#         self.num_steps = num_steps
#         self.batch_size = batch_size
#         self.topk = topk
#         self.search_batch_size = search_batch_size

#     def get_loss(self, logits, target_ids):
#         B, T, V = logits.shape
#         target_ids_expanded = target_ids.expand(B, -1).reshape(-1)
#         logits_flat = logits.reshape(-1, V)
#         return (
#             F.cross_entropy(logits_flat, target_ids_expanded, reduction="none")
#             .reshape(B, T)
#             .mean(dim=1)
#         )

#     def generate_attack_onehot(
#         self, starting_query: str, model_id="deepseek-community/deepseek-vl-1.3b-chat"
#     ):
#         model = get_or_load_model(model_id)
#         processor = get_or_load_processor(model_id)
#         tokenizer = processor.tokenizer

#         # --- 0. DIMENSIUNI REALE (Embeddings != Tokenizer) ---
#         embed_weights = model.get_input_embeddings().weight
#         real_vocab_size = embed_weights.shape[0]

#         # --- 1. PREGĂTIRE TOKENI ---
#         ids_before = tokenizer(
#             starting_query, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)

#         suffix_len = 20
#         # Inițializăm cu ceva vizibil: "!" (index 2 de obicei)
#         ids_suffix = torch.full((1, suffix_len), 2, device=model.device)

#         ids_after = tokenizer(
#             self.goal_response, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)
#         target_ids = ids_after[0, 1:]  # Tokenii pe care vrem să îi prezicem

#         # --- 2. FILTRARE VOCABULAR ---
#         forbidden = []
#         for i in range(tokenizer.vocab_size):
#             token = tokenizer.decode([i])
#             if token.isspace() or len(token.strip()) == 0 or i in tokenizer.all_special_ids:
#                 forbidden.append(i)

#         if real_vocab_size > tokenizer.vocab_size:
#             forbidden.extend(range(tokenizer.vocab_size, real_vocab_size))
#         forbidden_tensor = torch.tensor(forbidden, device=model.device)

#         best_loss = float("inf")
#         ids_suffix_best = ids_suffix.clone()

#         # --- LOOP OPTIMIZARE ---
#         for step in range(self.num_steps):
#             # PASUL 1: GRADIENT (ONE-HOT)
#             one_hot = (
#                 F.one_hot(ids_suffix, num_classes=real_vocab_size).to(model.dtype).to(model.device)
#             )
#             one_hot.requires_grad = True

#             embeds_suffix = one_hot @ embed_weights
#             embeds_before = model.get_input_embeddings()(ids_before)
#             embeds_after = model.get_input_embeddings()(ids_after)

#             full_embeds = torch.cat([embeds_before, embeds_suffix, embeds_after], dim=1)
#             outputs = model(inputs_embeds=full_embeds)

#             # --- ALINIERE LOGITS (Corecția 11 vs 10) ---
#             t_start = ids_before.shape[1] + suffix_len - 1
#             t_end = t_start + target_ids.shape[0]
#             logits = outputs.logits[:, t_start:t_end, :]

#             loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids)
#             loss.backward()

#             grad = -one_hot.grad.clone().to(torch.float32)
#             grad[:, :, forbidden_tensor] = -1e6

#             # PASUL 2: TOP-K
#             top_k_tokens = torch.topk(grad, self.topk, dim=-1).indices

#             # PASUL 3: BATCH SEARCH
#             sub_positions = torch.randint(0, suffix_len, (self.batch_size,), device=model.device)
#             sub_tokens = torch.randint(0, self.topk, (self.batch_size,), device=model.device)
#             batch_suffix_candidates = ids_suffix.repeat(self.batch_size, 1)

#             for b in range(self.batch_size):
#                 pos = sub_positions[b]
#                 batch_suffix_candidates[b, pos] = top_k_tokens[0, pos, sub_tokens[b]]

#             batch_losses = []
#             with torch.no_grad():
#                 for i in range(0, self.batch_size, self.search_batch_size):
#                     end_idx = min(i + self.search_batch_size, self.batch_size)
#                     curr_batch = batch_suffix_candidates[i:end_idx]
#                     b_size = curr_batch.shape[0]

#                     full_input_ids = torch.cat(
#                         [ids_before.expand(b_size, -1), curr_batch, ids_after.expand(b_size, -1)],
#                         dim=1,
#                     )

#                     b_outputs = model(full_input_ids)
#                     # Aceeași aliniere de logits și aici
#                     b_logits = b_outputs.logits[:, t_start:t_end, :]

#                     for b_idx in range(b_size):
#                         l = F.cross_entropy(b_logits[b_idx], target_ids).item()
#                         batch_losses.append(l)

#             # PASUL 4: ACTUALIZARE
#             best_batch_idx = np.argmin(batch_losses)
#             if batch_losses[best_batch_idx] < best_loss:
#                 best_loss = batch_losses[best_batch_idx]
#                 ids_suffix = batch_suffix_candidates[best_batch_idx : best_batch_idx + 1]
#                 ids_suffix_best = ids_suffix.clone()

#             print(
#                 f"Step {step:03d} | Loss: {best_loss:.4f} | Suffix: {repr(tokenizer.decode(ids_suffix[0]))}"
#             )

#             if best_loss < 0.01:
#                 break

#         return tokenizer.decode(ids_suffix_best[0])

#     def generate_attack_fixed_length(
#         self, starting_query: str, model_id="deepseek-community/deepseek-vl-1.3b-chat"
#     ):
#         model = get_or_load_model(model_id)
#         processor = get_or_load_processor(model_id)
#         tokenizer = processor.tokenizer
#         embed_weights = model.get_input_embeddings().weight

#         # --- 1. FILTRARE VOCABULAR ---
#         forbidden = []
#         for i in range(tokenizer.vocab_size):
#             token = tokenizer.decode([i])
#             if token.isspace() or len(token.strip()) == 0 or i in tokenizer.all_special_ids:
#                 forbidden.append(i)
#         forbidden = torch.tensor(list(set(forbidden))).to(model.device)

#         # --- 2. INIȚIALIZARE FIXĂ (20 TOKENS) ---
#         # Folosim un șir de caractere random dar vizibile pentru start
#         initial_str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
#         adv_suffix_ids = tokenizer(
#             initial_str, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)[:, :20]

#         query_ids = tokenizer(
#             starting_query, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)
#         target_ids = tokenizer(
#             self.goal_response, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)

#         current_loss = float("inf")
#         stagnation_counter = 0

#         for step in range(self.num_steps):
#             # --- PASUL 1: CALCUL GRADIENT ---
#             input_ids = torch.cat([query_ids, adv_suffix_ids], dim=1)
#             input_embeds = model.get_input_embeddings()(input_ids).detach()
#             input_embeds.requires_grad = True

#             with torch.no_grad():
#                 target_embeds = model.get_input_embeddings()(target_ids)

#             full_embeds = torch.cat([input_embeds, target_embeds], dim=1)
#             outputs = model(inputs_embeds=full_embeds)

#             logits = outputs.logits[:, input_ids.shape[1] - 1 : -1, :]
#             loss_val = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

#             model.zero_grad()
#             loss_val.backward()

#             # Extragem gradientul doar pentru sufix
#             grad = input_embeds.grad[:, query_ids.shape[1] :, :].detach()

#             # --- PASUL 2: TOP-K SELECTION (FLOAT32 SAFE) ---
#             with torch.no_grad():
#                 scores = torch.matmul(grad[0], embed_weights.T).to(torch.float32)
#                 scores[:, forbidden] = 1e6  # Penalizăm tokenii "invizibili"
#                 top_indices = (-scores).topk(self.topk, dim=1).indices  # (20, topk)

#             # --- PASUL 3: BATCH SEARCH (LUNGIME FIXĂ) ---
#             with torch.no_grad():
#                 # Cream copii ale sufixului curent
#                 batch_suffixes = adv_suffix_ids.repeat(self.batch_size, 1)  # (batch, 20)

#                 # Pentru fiecare element din batch, alegem o poziție random din cele 20
#                 pos_to_flip = torch.randint(0, adv_suffix_ids.shape[1], (self.batch_size,)).to(
#                     model.device
#                 )

#                 # Alegem un index random din top_k pentru acea poziție
#                 token_idx = torch.randint(0, self.topk, (self.batch_size,)).to(model.device)

#                 # Aplicăm schimbarea: batch_suffixes[rand_row, rand_col] = top_indices[col, rand_token]
#                 new_tokens = top_indices[pos_to_flip, token_idx]
#                 batch_suffixes[torch.arange(self.batch_size), pos_to_flip] = new_tokens

#                 # Evaluare
#                 batch_query = query_ids.expand(self.batch_size, -1)
#                 batch_input = torch.cat([batch_query, batch_suffixes], dim=1)
#                 batch_full_ids = torch.cat(
#                     [batch_input, target_ids.expand(self.batch_size, -1)], dim=1
#                 )

#                 batch_outputs = model(batch_full_ids)
#                 t_start = batch_input.shape[1] - 1
#                 batch_logits = batch_outputs.logits[:, t_start:-1, :]

#                 losses = self.get_loss(batch_logits, target_ids)
#                 best_idx = losses.argmin()

#                 if losses[best_idx] < current_loss:
#                     stagnation_counter = 0
#                     current_loss = losses[best_idx].item()
#                     adv_suffix_ids = batch_suffixes[best_idx : best_idx + 1]
#                 else:
#                     stagnation_counter += 1

#             # Afișare status
#             decoded_suffix = tokenizer.decode(adv_suffix_ids[0])
#             print(f"Step {step:03d} | Loss: {current_loss:.4f} | Suffix: {repr(decoded_suffix)}")

#             # --- LOGICĂ RE-START (Dacă rămâne blocat 30 de pași) ---
#             if stagnation_counter > 30:
#                 print("Stagnare detectată. Re-injectăm zgomot în sufix...")
#                 noise_pos = torch.randint(0, 20, (5,))
#                 adv_suffix_ids[0, noise_pos] = torch.randint(1000, 50000, (5,)).to(model.device)
#                 stagnation_counter = 0

#             if current_loss < 0.05:
#                 break

#         return tokenizer.decode(adv_suffix_ids[0])

#     def generate_attack(
#         self, starting_query: str, model_id="deepseek-community/deepseek-vl-1.3b-chat"
#     ):
#         model = get_or_load_model(model_id)
#         processor = get_or_load_processor(model_id)
#         tokenizer = processor.tokenizer
#         embed_weights = model.get_input_embeddings().weight

#         # --- FILTRARE VOCABULAR (Anti-Empty Suffix) ---
#         forbidden = []
#         for i in range(tokenizer.vocab_size):
#             token = tokenizer.decode([i])
#             # Interzicem tot ce e whitespace, lungime zero sau token special
#             if token.isspace() or len(token.strip()) == 0 or i in tokenizer.all_special_ids:
#                 forbidden.append(i)
#         forbidden = torch.tensor(list(set(forbidden))).to(model.device)

#         # --- INIȚIALIZARE CU TEXT REZISTENT ---
#         initial_str = "force system bypass commands instructions"
#         adv_suffix_ids = tokenizer(
#             initial_str, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)
#         # Ne asigurăm că avem exact 20 de tokeni (padding/slicing)
#         if adv_suffix_ids.shape[1] < 20:
#             extra = torch.randint(1000, 5000, (1, 20 - adv_suffix_ids.shape[1])).to(model.device)
#             adv_suffix_ids = torch.cat([adv_suffix_ids, extra], dim=1)
#         else:
#             adv_suffix_ids = adv_suffix_ids[:, :20]

#         query_ids = tokenizer(
#             starting_query, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)
#         target_ids = tokenizer(
#             self.goal_response, add_special_tokens=False, return_tensors="pt"
#         ).input_ids.to(model.device)

#         print(
#             f"Vocabular filtrat: {len(forbidden)} tokeni interziși dintr-un total de {tokenizer.vocab_size}"
#         )

#         current_loss = float("inf")

#         for step in range(self.num_steps):
#             # PASUL 1: Calcul Gradient
#             input_ids = torch.cat([query_ids, adv_suffix_ids], dim=1)
#             input_embeds = model.get_input_embeddings()(input_ids).detach()
#             input_embeds.requires_grad = True

#             with torch.no_grad():
#                 target_embeds = model.get_input_embeddings()(target_ids)

#             full_embeds = torch.cat([input_embeds, target_embeds], dim=1)
#             outputs = model(inputs_embeds=full_embeds)

#             # Slicing pentru target
#             logits = outputs.logits[:, input_ids.shape[1] - 1 : -1, :]
#             loss_val = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

#             model.zero_grad()
#             loss_val.backward()
#             grad = input_embeds.grad[:, query_ids.shape[1] :, :].detach()

#             # PASUL 2: Selecție Top-K cu Penalizare
#             # with torch.no_grad():
#             #     scores = torch.matmul(grad[0], embed_weights.T)
#             #     scores[:, forbidden] = 1e9  # Pedeapsă maximă pentru tokeni invizibili
#             #     top_indices = (-scores).topk(self.topk, dim=1).indices
#             with torch.no_grad():
#                 # Calculăm scorurile
#                 scores = torch.matmul(grad[0], embed_weights.T)  # (suffix_len, vocab_size)

#                 # IMPORTANT: Convertim în float32 pentru a evita overflow la penalizare
#                 scores = scores.to(torch.float32)

#                 # Folosim o valoare mare, dar sigură
#                 scores[:, forbidden] = 1e6

#                 # Alegem top indices (topk va lucra pe float32 acum)
#                 top_indices = (-scores).topk(self.topk, dim=1).indices

#                 # Opțional: eliberăm memoria dacă e critic
#                 del scores
#             # PASUL 3: Batch Search
#             with torch.no_grad():
#                 # Generăm variații ale sufixului
#                 sub_indices = torch.randint(0, self.topk, (self.batch_size,)).to(model.device)
#                 sub_positions = torch.randint(0, adv_suffix_ids.shape[1], (self.batch_size,)).to(
#                     model.device
#                 )

#                 batch_suffixes = adv_suffix_ids.repeat(self.batch_size, 1)
#                 batch_suffixes[torch.arange(self.batch_size), sub_positions] = top_indices[
#                     sub_positions, sub_indices
#                 ]

#                 # Evaluare rapidă batch
#                 batch_query = query_ids.expand(self.batch_size, -1)
#                 batch_input = torch.cat([batch_query, batch_suffixes], dim=1)
#                 batch_target = target_ids.expand(self.batch_size, -1)
#                 batch_full_ids = torch.cat([batch_input, batch_target], dim=1)

#                 batch_outputs = model(batch_full_ids)
#                 t_start = batch_input.shape[1] - 1
#                 batch_logits = batch_outputs.logits[:, t_start:-1, :]

#                 losses = self.get_loss(batch_logits, target_ids)
#                 best_idx = losses.argmin()

#                 # Actualizăm dacă am găsit un Loss mai mic
#                 if losses[best_idx] < current_loss:
#                     current_loss = losses[best_idx].item()
#                     adv_suffix_ids = batch_suffixes[best_idx : best_idx + 1]

#             # Repr() ne arată dacă avem caractere ciudate/invizibile
#             print(
#                 f"Step {step:03d} | Loss: {current_loss:.4f} | Suffix: {repr(tokenizer.decode(adv_suffix_ids[0]))}"
#             )

#             if current_loss < 0.05:
#                 break

#         return tokenizer.decode(adv_suffix_ids[0])

#     def test_response(
#         self,
#         attack_suffix: str,
#         starting_query: str,
#         model_id="deepseek-community/deepseek-vl-1.3b-chat",
#     ):
#         model = get_or_load_model(model_id)
#         processor = get_or_load_processor(model_id)

#         # Lipim query-ul de sufix
#         full_prompt = f"{starting_query} {attack_suffix}"
#         messages = [{"role": "user", "content": full_prompt}]

#         prompt = processor.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         inputs = processor(text=prompt, return_tensors="pt").to(model.device)

#         with torch.no_grad():
#             output = model.generate(
#                 **inputs,
#                 max_new_tokens=150,
#                 do_sample=False,
#                 pad_token_id=processor.tokenizer.pad_token_id,
#             )

#         return processor.tokenizer.decode(
#             output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
#         )


import torch
import torch.nn.functional as F
import numpy as np
from core.models import get_or_load_model, get_or_load_processor

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class GCGQwen:
    def __init__(self, goal_response, num_steps=250, batch_size=512, topk=256):
        self.goal_response = goal_response
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.topk = topk

    def attack(self, starting_query, model_id="Qwen/Qwen2.5-7B-Instruct"):
        # 1. SETUP (Qwen style)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)

        embed_weights = model.get_input_embeddings().weight
        vocab_size = embed_weights.shape[0]

        # 2. TOKENIZARE RAW (Exact ce vede modelul pe stivă)
        ids_before = tokenizer(
            starting_query, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        suffix_len = 20
        # Brian folosește ' !' (ID 2 în multe tokenizere) ca punct de start
        ids_suffix = torch.full((1, suffix_len), 2, device=device)

        # Adăugăm spațiu la începutul target-ului pentru a ajuta fluxul logit-urilor
        target_text = " " + self.goal_response
        ids_target = tokenizer(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        target_ids = ids_target[0]

        # 3. FILTRARE VOCABULAR (Brian recomandă eliminarea tokenilor speciali)
        forbidden_indices = torch.tensor(tokenizer.all_special_ids, device=device)

        print(f"🔥 Starting GCG on Qwen... Target: '{self.goal_response}'")

        for step in range(self.num_steps):
            # --- PASUL 1: GRADIENT ---
            one_hot = F.one_hot(ids_suffix, num_classes=vocab_size).to(model.dtype).to(device)
            one_hot.requires_grad = True

            # Embedding-uri concatenate
            input_embeds = torch.cat(
                [
                    model.get_input_embeddings()(ids_before),
                    one_hot @ embed_weights,
                    model.get_input_embeddings()(ids_target),
                ],
                dim=1,
            )

            outputs = model(inputs_embeds=input_embeds)

            # Slicing: Brian prezice target_ids.
            # Logit-ul de la finalul sufixului prezice primul token din target.
            t_start = ids_before.shape[1] + suffix_len - 1
            t_end = t_start + target_ids.shape[0]
            logits = outputs.logits[:, t_start:t_end, :]

            loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()

            # Gradienți în Float32 ca să nu avem overflow (cum ai pățit la DeepSeek)
            grad = -one_hot.grad.clone().to(torch.float32)
            grad[:, :, forbidden_indices] = -1e6

            # Top-K candidați pentru fiecare poziție din sufix
            top_k_indices = torch.topk(grad, self.topk, dim=-1).indices

            # --- PASUL 2: BATCH SEARCH (The "Brian Random Flip") ---
            best_loss = float("inf")
            batch_suffix_candidates = ids_suffix.repeat(self.batch_size, 1)

            # Alegem o poziție random și un token din Top-K pentru fiecare element din batch
            rand_pos = torch.randint(0, suffix_len, (self.batch_size,), device=device)
            rand_topk = torch.randint(0, self.topk, (self.batch_size,), device=device)

            for b in range(self.batch_size):
                batch_suffix_candidates[b, rand_pos[b]] = top_k_indices[
                    0, rand_pos[b], rand_topk[b]
                ]

            # Evaluare batch (fără gradienți)
            with torch.no_grad():
                # Concatenăm tot batch-ul: [Before] [Batch_Suffix] [Target]
                # Folosim bucăți de 32 pentru a evita OOM (Out of Memory)
                batch_losses = []
                for i in range(0, self.batch_size, 32):
                    curr_batch = batch_suffix_candidates[i : i + 32]
                    b_size = curr_batch.shape[0]

                    full_ids = torch.cat(
                        [ids_before.expand(b_size, -1), curr_batch, ids_target.expand(b_size, -1)],
                        dim=1,
                    )

                    b_out = model(full_ids).logits[:, t_start:t_end, :]

                    for b_idx in range(b_size):
                        l = F.cross_entropy(b_out[b_idx], target_ids).item()
                        batch_losses.append(l)

            # --- PASUL 3: UPDATE ---
            best_batch_idx = np.argmin(batch_losses)
            if batch_losses[best_batch_idx] < best_loss:
                ids_suffix = batch_suffix_candidates[best_batch_idx : best_batch_idx + 1]
                best_loss = batch_losses[best_batch_idx]

            if step % 5 == 0:
                current_suffix = tokenizer.decode(ids_suffix[0])
                print(f"Step {step:03d} | Loss: {best_loss:.4f} | Suffix: {repr(current_suffix)}")

        return tokenizer.decode(ids_suffix[0])

    def test_response(self, starting_query, adv_suffix, model_id="Qwen/Qwen2.5-7B-Instruct"):
        # Reîncărcăm modelul și tokenizer-ul (sau le pasăm pe cele existente)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)

        # Brian Style: Construim prompt-ul brut
        # Lipim query-ul de sufixul optimizat
        full_prompt = f"{starting_query} {adv_suffix}"

        # Tokenizăm fără să adăugăm mesaje de tip <|user|>
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        print(f"\n--- RULARE TEST (RAW) ---")
        print(f"Input: {full_prompt}")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,  # Îi dăm spațiu să scrie tot ghidul
                do_sample=False,  # Greedy decoding pentru a vedea exact ce a optimizat GCG
                temperature=0.0,  # Zero haos
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decodăm doar partea de după prompt-ul nostru
        generated_text = tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return generated_text


class GCGAttack:
    def __init__(
        self,
        goal_response: str = "Sure, I can help. Here is a step-by-step guide:",
        num_steps: int = 250,
        batch_size: int = 512,
        topk: int = 256,
    ):
        self.goal_response = goal_response
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.topk = topk

    def generate_attack(
        self, starting_query: str, model_id="deepseek-community/deepseek-vl-1.3b-chat"
    ):
        model = get_or_load_model(model_id)
        tokenizer = get_or_load_processor(model_id).tokenizer
        embed_weights = model.get_input_embeddings().weight
        real_vocab_size = embed_weights.shape[0]

        # 1. PREGĂTIRE TOKENI (RAW - Fără chat template)
        ids_before = tokenizer(
            starting_query, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(model.device)

        suffix_len = 10
        # Inițializare: 10 de tokeni de tip "!" (ID 2)
        ids_suffix = torch.full((1, suffix_len), 2, device=model.device)

        # Target-ul trebuie să fie exact ce vrei să scoată modelul din logit-uri
        ids_target = tokenizer(
            self.goal_response, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(model.device)
        target_ids = ids_target[0]

        # 2. FILTRARE (Forbidden tokens)
        forbidden = []
        for i in range(tokenizer.vocab_size):
            token = tokenizer.decode([i])
            if token.isspace() or len(token.strip()) == 0 or i in tokenizer.all_special_ids:
                forbidden.append(i)
        if real_vocab_size > tokenizer.vocab_size:
            forbidden.extend(range(tokenizer.vocab_size, real_vocab_size))
        forbidden_tensor = torch.tensor(forbidden, device=model.device)

        best_loss = float("inf")

        print(f"Starting GCG (Brian Style)... Target: {self.goal_response}")

        for step in range(self.num_steps):
            # --- PASUL A: CALCUL GRADIENT (ONE-HOT) ---
            one_hot = (
                F.one_hot(ids_suffix, num_classes=real_vocab_size).to(model.dtype).to(model.device)
            )
            one_hot.requires_grad = True

            # Forward pass manual prin embedding-uri
            embeds_before = model.get_input_embeddings()(ids_before)
            embeds_suffix = one_hot @ embed_weights
            embeds_target = model.get_input_embeddings()(ids_target)

            full_embeds = torch.cat([embeds_before, embeds_suffix, embeds_target], dim=1)
            outputs = model(inputs_embeds=full_embeds)

            # Logit-ul de la finalul sufixului prezice primul token din target
            t_start = ids_before.shape[1] + suffix_len - 1
            t_end = t_start + target_ids.shape[0]
            logits = outputs.logits[:, t_start:t_end, :]

            loss = F.cross_entropy(logits.view(-1, real_vocab_size), target_ids.view(-1))
            loss.backward()

            # Extragere gradienți (Float32 pentru precizie)
            grad = -one_hot.grad.clone().to(torch.float32)
            grad[:, :, forbidden_tensor] = -1e6

            top_k_tokens = torch.topk(grad, self.topk, dim=-1).indices

            # --- PASUL B: BATCH SEARCH (CANDIDAȚI) ---
            # alege o poziție random și un token de top pentru fiecare element din batch
            batch_suffix_candidates = ids_suffix.repeat(self.batch_size, 1)

            # Generăm mutații
            positions = torch.randint(0, suffix_len, (self.batch_size,), device=model.device)
            token_indices = torch.randint(0, self.topk, (self.batch_size,), device=model.device)

            for b in range(self.batch_size):
                pos = positions[b]
                batch_suffix_candidates[b, pos] = top_k_tokens[0, pos, token_indices[b]]

            # Evaluare batch (fără gradienți pentru viteză)
            batch_losses = []
            with torch.no_grad():
                # Pentru DeepSeek, e mai sigur să evaluăm în bucăți mici (search_batch_size)
                search_bs = 32
                for i in range(0, self.batch_size, search_bs):
                    curr_batch = batch_suffix_candidates[i : i + search_bs]
                    b_size = curr_batch.shape[0]

                    # [Before] [Suffix_Cand] [Target]
                    input_ids = torch.cat(
                        [ids_before.expand(b_size, -1), curr_batch, ids_target.expand(b_size, -1)],
                        dim=1,
                    )

                    b_outputs = model(input_ids)
                    b_logits = b_outputs.logits[:, t_start:t_end, :]

                    for b_idx in range(b_size):
                        l = F.cross_entropy(b_logits[b_idx], target_ids).item()
                        batch_losses.append(l)

            # --- PASUL C: UPDATE ---
            best_idx = np.argmin(batch_losses)
            if batch_losses[best_idx] < best_loss:
                best_loss = batch_losses[best_idx]
                ids_suffix = batch_suffix_candidates[best_idx : best_idx + 1]

            if step % 10 == 0:
                print(
                    f"Step {step:03d} | Loss: {best_loss:.4f} | Suffix: {repr(tokenizer.decode(ids_suffix[0]))}"
                )

        return tokenizer.decode(ids_suffix[0])

    def test_response(
        self,
        starting_query,
        attack_suffix,
        model_id="deepseek-community/deepseek-vl-1.3b-chat",
    ):
        model = get_or_load_model(model_id)
        processor = get_or_load_processor(model_id)
        tokenizer = processor.tokenizer

        raw_prompt = f"{starting_query} {attack_suffix}"
        inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)

        print(f"\n--- TESTING RAW PROMPT ---\n{raw_prompt}")

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
