import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Literal


class SimilarityEvaluator:
    """
    Similarity between 2 string sentences: 0-1
    """

    def __init__(
        self,
        embedding_type: Literal["bert", "count", "tfidf"] = "bert",
        metric: Literal["cosine", "euclidean"] = "cosine",
    ):
        """TODO: frozen paraphrase encoder
        https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        """
        self.embedding_type = embedding_type
        self.metric = metric

        if self.embedding_type == "bert":
            model_name = "prajjwal1/bert-tiny"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)

            # Detect device: Move model to GPU if available, else CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

        elif self.embedding_type in ["tfidf", "count"]:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

                self.vectorizer = (
                    TfidfVectorizer() if embedding_type == "tfidf" else CountVectorizer()
                )
            except ImportError:
                raise ImportError("Install sklearn for 'tfidf' or 'count'.")

    def _manual_cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot / (norm_v1 * norm_v2))

    def _get_bert_embedding(self, sentence: str) -> np.ndarray:
        # 1. Tokenize (creates tensors on CPU by default)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        # 2. MOVE ARGUMENTS TO THE STACK (Same device as model)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Move back to CPU to convert to NumPy for the manual cosine math
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def evaluate(self, s1: str, s2: str) -> float:
        if self.embedding_type == "bert":
            v1 = self._get_bert_embedding(s1)
            v2 = self._get_bert_embedding(s2)
        else:
            vectors = self.vectorizer.fit_transform([s1, s2]).toarray()
            v1, v2 = vectors[0], vectors[1]

        if self.metric == "cosine":
            return self._manual_cosine(v1, v2)

        dist = np.linalg.norm(v1 - v2)
        return 1 / (1 + dist)


# --- Usage ---
# evaluator = SimilarityEvaluator(embedding_type='bert')
# print(evaluator.evaluate("Hello world", "Hi there"))


class DummySimilarityEvaluator(SimilarityEvaluator):
    def __init__(self, range_eval: list[float]):
        self.range_eval = range_eval
        self.i = -1

    def evaluate(self, s1, s2):
        self.i += 1
        return self.range_eval[self.i % len(self.range_eval)]
