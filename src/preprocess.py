"""Dataset preprocessing and loading for GSM8K reasoning experiments."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset


def load_gsm8k_dataset(
    split: str = "test",
    subset_size: Optional[int] = None,
    seed: int = 42,
    cache_dir: str = ".cache",
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset for math reasoning evaluation.

    Args:
        split: Dataset split ('train' or 'test')
        subset_size: Number of examples to sample (None for full dataset)
        seed: Random seed for reproducible sampling
        cache_dir: Directory to cache downloaded datasets

    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    # Load GSM8K from HuggingFace datasets
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list of dicts
    examples = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]

        # Extract numerical answer from GSM8K format
        # GSM8K answers are in format: "explanation\n#### numerical_answer"
        numerical_answer = extract_numerical_answer(answer)

        examples.append(
            {
                "question": question,
                "answer": numerical_answer,
                "full_answer": answer,  # Keep full answer for reference
            }
        )

    # Sample subset if requested
    if subset_size is not None and subset_size < len(examples):
        import random

        random.seed(seed)
        examples = random.sample(examples, subset_size)

    return examples


def extract_numerical_answer(answer_text: str) -> str:
    """
    Extract numerical answer from GSM8K answer format.

    GSM8K answers are in format: "explanation\n#### numerical_answer"

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Numerical answer as string
    """
    # GSM8K uses #### to separate explanation from answer
    if "####" in answer_text:
        numerical_answer = answer_text.split("####")[-1].strip()
    else:
        # Fallback: try to extract number from end of text
        numbers = re.findall(r"[\d,]+(?:\.\d+)?", answer_text)
        numerical_answer = numbers[-1] if numbers else answer_text.strip()

    # Remove commas from numbers
    numerical_answer = numerical_answer.replace(",", "")

    return numerical_answer


def normalize_answer(answer: str) -> str:
    """
    Normalize numerical answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    # Remove whitespace
    answer = answer.strip()

    # Remove commas
    answer = answer.replace(",", "")

    # Remove dollar signs and other currency symbols
    answer = answer.replace("$", "").replace("€", "").replace("£", "")

    # Try to convert to float and back to remove trailing zeros
    try:
        num = float(answer)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        return str(num)
    except (ValueError, AttributeError):
        return answer


# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: Answer extraction is failing, returning only commas "," instead of actual numbers
# [CAUSE]: The regex pattern [\d,]+ matches commas alone (e.g., "Therefore," → ",").
#          Pattern 2 in fallback "therefore[,:]?\s*([\d,]+)" was matching "Therefore," and capturing only the comma.
# [FIX]: Changed all number-matching patterns from [\d,]+ to \d[\d,]* to require at least one digit
#
# [OLD CODE]:
# def extract_answer_from_response(response: str, pattern: Optional[str] = None) -> str:
#     if pattern:
#         match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
#         if match:
#             return match.group(1).strip()
#     patterns = [
#         r"(?i)(?:final answer|the answer is|answer)\s*:?\s*([\d,]+(?:\.\d+)?)",
#         r"(?i)####\s*([\d,]+(?:\.\d+)?)",
#         r"(?i)therefore[,:]?\s*([\d,]+(?:\.\d+)?)",
#     ]
#     for p in patterns:
#         match = re.search(p, response, re.IGNORECASE | re.MULTILINE)
#         if match:
#             return match.group(1).strip()
#     numbers = re.findall(r"[\d,]+(?:\.\d+)?", response)
#     if numbers:
#         return numbers[-1].strip()
#     return ""
#
# [NEW CODE]:
def extract_answer_from_response(response: str, pattern: Optional[str] = None) -> str:
    """
    Extract final answer from model response.

    Args:
        response: Model's full response text
        pattern: Regex pattern to extract answer (optional)

    Returns:
        Extracted answer string
    """
    if pattern:
        # Use provided regex pattern
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Fallback patterns (require at least one digit)
    patterns = [
        r"(?i)(?:final answer|the answer is|answer)\s*:?\s*(\d[\d,]*(?:\.\d+)?)",
        r"(?i)####\s*(\d[\d,]*(?:\.\d+)?)",
        r"(?i)therefore[,:]?\s*(\d[\d,]*(?:\.\d+)?)",
    ]

    for p in patterns:
        match = re.search(p, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Last resort: find last number in response (require at least one digit)
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", response)
    if numbers:
        return numbers[-1].strip()

    return ""


def count_reasoning_steps(response: str) -> int:
    """
    Count the number of reasoning steps in the response.

    Args:
        response: Model's response text

    Returns:
        Number of reasoning steps
    """
    # Count sentences or numbered steps
    sentences = re.split(r"[.!?]\n", response)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Also check for numbered steps
    numbered_steps = re.findall(r"^\d+[\.)]", response, re.MULTILINE)

    return max(len(sentences), len(numbered_steps))


def count_decisions(response: str) -> int:
    """
    Count critical decisions in DFSV-CoT response.

    Args:
        response: Model's response text

    Returns:
        Number of decisions
    """
    # Look for [FACT], [RULE], [INFER] markers
    decisions = re.findall(r"\[(FACT|RULE|INFER)\]", response, re.IGNORECASE)
    return len(decisions)


def count_uncertain_decisions(response: str) -> int:
    """
    Count uncertain decisions in DFSV-CoT response.

    Args:
        response: Model's response text

    Returns:
        Number of uncertain decisions
    """
    uncertain = re.findall(r"\[UNCERTAIN\]", response, re.IGNORECASE)
    return len(uncertain)


def check_support_violations(response: str) -> int:
    """
    Check for support verification violations.

    Args:
        response: Model's response text

    Returns:
        Number of violations (simple heuristic)
    """
    # This is a simple heuristic - count if answer given without resolving UNCERTAIN
    has_uncertain = "[UNCERTAIN]" in response.upper()
    has_answer = re.search(r"(?i)(final answer|answer is)", response) is not None

    if has_uncertain and has_answer:
        # Check if there's a revision section
        has_revision = (
            re.search(r"(?i)(revision|resolve|correcting)", response) is not None
        )
        if not has_revision:
            return 1

    return 0


if __name__ == "__main__":
    # Test dataset loading
    examples = load_gsm8k_dataset(split="test", subset_size=5, seed=42)
    print(f"Loaded {len(examples)} examples")
    print("\nFirst example:")
    print(f"Question: {examples[0]['question']}")
    print(f"Answer: {examples[0]['answer']}")
