from datasets import Dataset
import numpy as np
import pandas as pd


def sanitize_reward(r: float) -> float:
    r = min(max(r, 0.0), 0.95)
    r += np.random.normal(0, 0.02)
    return float(np.clip(r, 0.0, 1.0))


def filter_dataset(dataset, min_reward=0.7):
    return [ex for ex in dataset if ex["reward"] >= min_reward]


class ReplayBuffer:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.buffer = []

    def add(self, samples):
        self.buffer.extend(samples)
        self.buffer = sorted(self.buffer, key=lambda x: x["reward"], reverse=True)[: self.max_size]

    def sample(self, k):
        if len(self.buffer) == 0:
            return []
        k = min(k, len(self.buffer))
        idx = np.random.choice(len(self.buffer), k, replace=False)
        return [self.buffer[i] for i in idx]
