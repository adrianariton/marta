import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import os


def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []
    
convos = read_jsonl('MORAAIQVB100.jsonl')
# 'eval_PEC7_phi-4-AWQ', 'eval_PEC7_Mixtral-8x22B-Instruct-v0.1-AWQ', 'eval_PEC7_Qwen2.5-32B-Instruct-AWQ', 'eval_PEC7_Qwen2.5-72B-Instruct-AWQ'
print(list(convos[0].keys()))
