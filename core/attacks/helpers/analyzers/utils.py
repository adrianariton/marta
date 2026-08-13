import nltk
import numpy as np
from wordfreq import word_frequency


def sort_by_inverse_frequency(word_list, lang="en"):
    """
    Orders a list of words from rarest to most common.
    All arguments are to the stack: (word_list, lang) -> sorted_list
    """
    return sorted(word_list, key=lambda w: word_frequency(w, lang))


def _tag_POS(sentence: str) -> list[tuple[str, str]]:
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return tagged


def tag_POS(sentence: str) -> list[tuple[str, str]]:
    """
    Utility to POS tag a sentence.
    Input: A raw string (pushed to function).
    Output: A list of (word, tag) tuples.
    """
    try:
        return _tag_POS(sentence)
    except LookupError:
        # Handle missing NLTK resources automatically
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("averaged_perceptron_tagger_eng")
        return _tag_POS(sentence)


def tag_POS_and_keep_useful_words(sentence: str) -> list[tuple[str, str]]:
    """
    Filters a sentence to return only Nouns, Verbs, Adjectives, and Adverbs.
    Input: Raw string.
    Output: Filtered list of (word, tag) tuples.
    """

    full_tagged_stack = tag_POS(sentence)
    content_prefixes = ("N", "V", "J", "R")

    filtered_stack = [
        (word, tag) for word, tag in full_tagged_stack if tag.startswith(content_prefixes)
    ]

    return filtered_stack


import spacy
import spacy.util


def load_nlp_model(model_name: str = "en_core_web_md"):
    """
    Checks if a spaCy model is installed.
    If not, downloads it before returning the loaded nlp object.
    """
    if not spacy.util.is_package(model_name):
        print(f"--- Model {model_name} not found. Downloading to stack... ---")
        spacy.cli.download(model_name)

    return spacy.load(model_name)


def get_cosine_sim_matrix(list_a: list[str], list_b: list[str]) -> np.ndarray:
    """
    Rows: len(list_a)
    Cols: len(list_b)
    """
    nlp = load_nlp_model("en_core_web_md")

    vecs_a = np.array([nlp(w).vector for w in list_a])
    vecs_b = np.array([nlp(w).vector for w in list_b])

    norm_a = vecs_a / (np.linalg.norm(vecs_a, axis=1, keepdims=True) + 1e-9)
    norm_b = vecs_b / (np.linalg.norm(vecs_b, axis=1, keepdims=True) + 1e-9)

    return np.dot(norm_a, norm_b.T)
