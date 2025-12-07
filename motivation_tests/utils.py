"""
Utility functions for motivation tests
"""
import tiktoken
import numpy as np
from typing import List, Dict, Any
import json


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def estimate_token_cost(text: str) -> int:
    """Estimate token cost for a text"""
    return count_tokens(text)


def truncate_to_budget(text: str, budget: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token budget"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= budget:
        return text
    truncated_tokens = tokens[:budget]
    return encoding.decode(truncated_tokens)


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 95, 99]) -> Dict[int, float]:
    """Calculate percentiles for a list of values"""
    sorted_values = sorted(values)
    result = {}
    for p in percentiles:
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        result[p] = sorted_values[idx]
    return result

