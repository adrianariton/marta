from abc import ABC, abstractmethod
from typing import List
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


class TopicExtractor(ABC):
    """Base class for topic extraction models."""

    def __init__(self, n_topics: int = 5, n_top_words: int = 10):
        self.n_topics = n_topics
        self.n_top_words = n_top_words

    @abstractmethod
    def fit_transform(self, documents: List[str]) -> List[List[str]]:
        """
        Fit the model and extract topics from a list of documents.

        Args:
            documents: List of text documents.

        Returns:
            List of topics, where each topic is a list of top words.
        """
        pass

    @abstractmethod
    def assign(self, documents: List[str]) -> List[List[int]]:
        """
        Assign each document to its most likely topic.
        Must be called after fit_transform.

        Args:
            documents: List of text documents.

        Returns:
            List of topic indices, one per document.
        """
        pass

    def fit_transform_assign(self, documents: List[str]) -> tuple[List[List[str]], List[List[int]]]:
        """Fit, extract topics, and assign in one call."""
        topics = self.fit_transform(documents)
        assignments = self.assign(documents)
        return topics, assignments

    def _get_top_words(self, components, feature_names: List[str]) -> List[List[str]]:
        """Extract top words for each topic from model components."""
        topics = []
        for topic in components:
            top_indices = topic.argsort()[-self.n_top_words :][::-1]
            topics.append([feature_names[i] for i in top_indices])
        return topics


class LDATopicExtractor(TopicExtractor):
    """Topic extractor using Latent Dirichlet Allocation (LDA)."""

    def __init__(self, n_topics: int = 5, n_top_words: int = 10, random_state: int = 42):
        super().__init__(n_topics, n_top_words)
        self.random_state = random_state
        self.vectorizer = CountVectorizer(stop_words="english")
        self.model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)

    def fit_transform(self, documents: List[str]) -> List[List[str]]:
        X = self.vectorizer.fit_transform(documents)
        self.model.fit(X)
        feature_names = self.vectorizer.get_feature_names_out()
        return self._get_top_words(self.model.components_, feature_names)

    def assign(self, documents: List[str], threshold: float = 0.1) -> List[List[int]]:
        X = self.vectorizer.transform(documents)
        topic_distributions = self.model.transform(X)  # shape: (n_docs, n_topics)
        return [
            [i for i, score in enumerate(dist) if score >= threshold]
            for dist in topic_distributions
        ]


class NMFTopicExtractor(TopicExtractor):
    """Topic extractor using Non-negative Matrix Factorization (NMF)."""

    def __init__(self, n_topics: int = 5, n_top_words: int = 10, random_state: int = 42):
        super().__init__(n_topics, n_top_words)
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.model = NMF(n_components=n_topics, random_state=random_state)

    def fit_transform(self, documents: List[str]) -> List[List[str]]:
        X = self.vectorizer.fit_transform(documents)
        self.model.fit(X)
        feature_names = self.vectorizer.get_feature_names_out()
        return self._get_top_words(self.model.components_, feature_names)

    def assign(self, documents: List[str], threshold: float = 0.1) -> List[List[int]]:
        X = self.vectorizer.transform(documents)
        topic_distributions = self.model.transform(X)  # shape: (n_docs, n_topics)
        return [
            [i for i, score in enumerate(dist) if score >= threshold]
            for dist in topic_distributions
        ]


class BERTopicExtractor(TopicExtractor):
    """Topic extractor using BERTopic (semantic, transformer-based)."""

    def __init__(
        self,
        n_topics: int = 5,
        n_top_words: int = 10,
        n_docs: int = 10,
        random_state: int = None,
        approximate_distribution_for_assignment: bool = True,
    ):
        super().__init__(n_topics, n_top_words)
        self.approximate_distribution_for_assignment = approximate_distribution_for_assignment
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN

            n_neighbors = max(2, min(5, n_docs - 1))
            n_components = max(2, min(5, n_docs - 2))
            min_cluster_size = max(2, min(5, n_docs // 2))
            min_samples = max(1, min(2, n_docs // 3))

            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                random_state=random_state,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                prediction_data=True,
            )
            vectorizer_model = CountVectorizer(stop_words="english")
            self.model = BERTopic(
                nr_topics=n_topics,
                top_n_words=n_top_words,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
            )
        except ImportError:
            raise ImportError("Install bertopic with: pip install bertopic")

    def fit_transform(self, documents: List[str]) -> List[List[str]]:
        if len(documents) < 5:
            raise ValueError(f"BERTopicExtractor needs at least 5 documents, got {len(documents)}")

        self._fitted_topics, _ = self.model.fit_transform(documents)
        topic_ids = set(self._fitted_topics) - {-1}  # -1 is the outlier topic in BERTopic
        result = []
        for topic_id in sorted(topic_ids):
            words = [word for word, _ in self.model.get_topic(topic_id)]
            result.append(words)
        return result

    def assign(self, documents: List[str], threshold: float = 0.1) -> List[List[int]]:
        topic_distr, _ = self.model.approximate_distribution(documents)  # (n_docs, n_topics)
        assignments = [
            [i for i, score in enumerate(dist) if score >= threshold] for dist in topic_distr
        ]
        return assignments

    def fit_transform_assign__fast(self, documents: List[str], threshold: float = 0.1):
        topics = self.fit_transform(documents)  # already fitted, stored in self._fitted_topics
        assignments = [[t] if t != -1 else [0] for t in self._fitted_topics]
        return topics, assignments

    def fit_transform_assign(
        self, documents: List[str], threshold: float = 0.1
    ) -> tuple[List[List[str]], List[List[int]]]:
        if not self.approximate_distribution_for_assignment:
            return self.fit_transform_assign__fast(documents, threshold)
        topics = self.fit_transform(documents)
        topic_distr, _ = self.model.approximate_distribution(documents)  # (n_docs, n_topics)
        assignments = [
            [i for i, score in enumerate(dist) if score >= threshold] for dist in topic_distr
        ]
        return topics, assignments
