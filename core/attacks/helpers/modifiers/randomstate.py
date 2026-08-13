import random
import re
from core.attacks.helpers.modifiers.base import QueryModifier
import warnings
from typing import Optional


class RSMType:
    def __init__(self):
        pass


class RSMTypeLetterInserter(RSMType):
    def __init__(self, alphabet: str = "abcdefghijklmnopqrstuvwxyz", density: float = 0.1):
        self.alphabet = alphabet
        self.density = density


class RSMTypeWordShuffler(RSMType):
    def __init__(self, n_words: int = 2, probability: float = 0.5):
        self.n_words = n_words
        self.probability = probability


class RSMTypeSynonymSwapper(RSMType):
    def __init__(self, skip_words: set = None):
        self.skip_words = skip_words or {"the", "and", "is", "of", "to", "in", "a", "with", "it"}


class RSMTypeLeaveLikeItIs(RSMType):
    def __init__(self):
        super().__init__()


class RandomStateModifier(QueryModifier):
    def __init__(self, RSMType: Optional[RSMType] = None):
        super().__init__()
        self.RSMType = RSMType or RSMTypeLeaveLikeItIs()

    def modify(self, q: str, n: int, random_state: random.Random) -> list[str]:
        """
        Processes the query. Arguments are treated as a stack:
        [q (str), n (int), random_state (Random)]
        """
        results = []

        for _ in range(n):
            if isinstance(self.RSMType, RSMTypeLetterInserter):
                modified = self._insert_letters(q, random_state)
            elif isinstance(self.RSMType, RSMTypeWordShuffler):
                modified = self._shuffle_words(q, random_state)
            elif isinstance(self.RSMType, RSMTypeSynonymSwapper):
                modified = self._swap_synonyms(q, random_state)
            else:
                modified = q
            results.append(modified)

        return results

    def _insert_letters(self, q: str, rs: random.Random) -> str:
        chars = list(q)
        num_to_insert = int(len(chars) * self.RSMType.density)
        for _ in range(num_to_insert):
            idx = rs.randint(0, len(chars))
            chars.insert(idx, rs.choice(self.RSMType.alphabet))
        return "".join(chars)

    def _shuffle_words(self, q: str, rs: random.Random) -> str:
        if rs.random() > self.RSMType.probability:
            return q

        words = q.split()
        if len(words) < self.RSMType.n_words:
            return q

        # Select n words to "pop" from the stack and rotate
        n = self.RSMType.n_words
        start_idx = rs.randint(0, len(words) - n)
        subset = words[start_idx : start_idx + n]

        # n/2 forward, n/2 backward logic
        mid = n // 2
        reordered = subset[mid:] + subset[:mid]

        words[start_idx : start_idx + n] = reordered
        return " ".join(words)

    def _swap_synonyms(self, q: str, rs: random.Random) -> str:
        # Dynamic NLTK loading
        try:
            import nltk
            from nltk.corpus import wordnet

            try:
                wordnet.ensure_loaded()
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
        except ImportError:
            warnings.warn("NLTK Not found, ignoring synonyms swapping.")
            return q

        words = q.split()
        new_words = []
        for word in words:
            clean_word = re.sub(r"[^\w]", "", word).lower()
            if clean_word in self.RSMType.skip_words:
                new_words.append(word)
                continue

            syns = wordnet.synsets(clean_word)
            if syns:
                # Filter for actual lemmas that aren't the word itself
                lemmas = [
                    l.name().replace("_", " ")
                    for s in syns
                    for l in s.lemmas()
                    if l.name() != clean_word
                ]
                if lemmas:
                    new_words.append(rs.choice(lemmas))
                    continue
            new_words.append(word)

        return " ".join(new_words)
