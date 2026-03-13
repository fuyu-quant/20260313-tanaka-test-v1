"""Data preprocessing for GSM8K dataset."""

import os
import re
from typing import Dict, List, Any
from datasets import load_dataset


def load_gsm8k(
    split: str = "test", subset_size: int = 200, cache_dir: str = ".cache"
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split (train/test)
        subset_size: Number of examples to load (for cost control)
        cache_dir: Cache directory for HuggingFace datasets

    Returns:
        List of examples with 'question' and 'answer' fields
    """
    # Load dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Take subset
    if subset_size and subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))

    # Convert to list of dicts
    examples = []
    for item in dataset:
        # GSM8K format: question is string, answer is string with "#### <number>" format
        question = item["question"]
        answer_text = item["answer"]

        # Extract numeric answer from "#### <number>" format
        numeric_answer = extract_numeric_answer(answer_text)

        examples.append(
            {
                "question": question,
                "answer": numeric_answer,
                "full_answer": answer_text,  # keep for reference
            }
        )

    return examples


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract the final numeric answer from GSM8K answer format.
    GSM8K answers end with "#### <number>"

    Args:
        answer_text: Full answer text

    Returns:
        Numeric answer as float
    """
    # Find the final answer after ####
    match = re.search(r"####\s*([0-9,.-]+)", answer_text)
    if match:
        # Remove commas and convert to float
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            print(f"Warning: Could not parse answer: {num_str}")
            return None
    else:
        print(f"Warning: No #### found in answer: {answer_text[:100]}")
        return None


def normalize_number(text: str) -> float:
    """
    Extract and normalize a number from text.
    Handles various formats: "42", "42.5", "$42", "42%", "42.5k", etc.

    Args:
        text: Text containing a number

    Returns:
        Normalized float, or None if no number found
    """
    if text is None:
        return None

    # Remove common symbols and whitespace
    text = str(text).strip()

    # Remove currency symbols, percent signs, etc.
    text = re.sub(r"[$,%]", "", text)

    # Handle "k" suffix (thousands)
    if "k" in text.lower():
        text = text.lower().replace("k", "")
        multiplier = 1000
    else:
        multiplier = 1

    # Extract the last number (most likely to be the final answer)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", "")) * multiplier
        except ValueError:
            return None

    return None


def check_answer_match(
    predicted: Any, ground_truth: Any, tolerance: float = 1e-4
) -> bool:
    """
    Check if predicted answer matches ground truth.

    Args:
        predicted: Predicted answer (number or text)
        ground_truth: Ground truth answer (number)
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if answers match, False otherwise
    """
    # Normalize both answers
    pred_num = normalize_number(str(predicted)) if predicted is not None else None
    gt_num = float(ground_truth) if ground_truth is not None else None

    if pred_num is None or gt_num is None:
        return False

    # Compare with tolerance
    return abs(pred_num - gt_num) < tolerance
