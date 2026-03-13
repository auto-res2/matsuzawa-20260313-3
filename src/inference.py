"""Inference execution for prompt-tuning experiments."""

import json
import sys
from pathlib import Path
from typing import Dict, List
import wandb
from omegaconf import DictConfig, OmegaConf

from src.model import create_model_from_config
from src.preprocess import (
    load_gsm8k_dataset,
    normalize_answer,
    extract_answer_from_response,
    count_reasoning_steps,
    count_decisions,
    count_uncertain_decisions,
    check_support_violations,
)


def run_inference(cfg: DictConfig) -> Dict:
    """
    Run inference for a single experiment configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with results and metrics
    """
    print(f"Starting inference for run: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")

    # Initialize WandB if enabled
    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        # Override project name for sanity/pilot modes
        project = cfg.wandb.project
        if cfg.mode == "sanity":
            project = f"{cfg.wandb.project}-sanity"
        elif cfg.mode == "pilot":
            project = f"{cfg.wandb.project}-pilot"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run initialized: {wandb.run.url}")

    # Load dataset
    print("Loading dataset...")
    subset_size = cfg.run.dataset.subset_size

    # Adjust subset size based on mode
    if cfg.mode == "sanity":
        subset_size = min(10, subset_size)  # 5-10 samples for sanity
    elif cfg.mode == "pilot":
        subset_size = max(50, int(subset_size * 0.2))  # 20% for pilot, min 50

    examples = load_gsm8k_dataset(
        split=cfg.run.dataset.split,
        subset_size=subset_size,
        seed=cfg.run.dataset.seed,
        cache_dir=cfg.cache_dir,
    )
    print(f"Loaded {len(examples)} examples")

    # Create model
    print("Creating model...")
    model = create_model_from_config(cfg.run.model)

    # Get prompt template
    prompt_template = cfg.run.method.prompt_template

    # Run inference
    print("Running inference...")
    results = []
    correct = 0
    total_steps = 0
    total_decisions = 0
    total_uncertain = 0
    total_violations = 0

    for i, example in enumerate(examples):
        # Format prompt
        prompt = prompt_template.format(question=example["question"])

        # Generate response
        try:
            response = model.generate(prompt)
        except Exception as e:
            print(f"Error generating response for example {i}: {e}")
            response = ""

        # Extract answer
        answer_pattern = cfg.run.inference.get("answer_extraction_pattern", None)
        predicted_answer = extract_answer_from_response(response, answer_pattern)

        # Normalize answers
        pred_normalized = normalize_answer(predicted_answer)
        gold_normalized = normalize_answer(example["answer"])

        # Check correctness
        is_correct = pred_normalized == gold_normalized
        if is_correct:
            correct += 1

        # Compute reasoning metrics
        num_steps = count_reasoning_steps(response)
        num_decisions = count_decisions(response)
        num_uncertain = count_uncertain_decisions(response)
        num_violations = check_support_violations(response)

        total_steps += num_steps
        total_decisions += num_decisions
        total_uncertain += num_uncertain
        total_violations += num_violations

        # Store result
        result = {
            "example_id": i,
            "question": example["question"],
            "gold_answer": example["answer"],
            "predicted_answer": predicted_answer,
            "response": response,
            "correct": is_correct,
            "reasoning_steps": num_steps,
            "decision_count": num_decisions,
            "uncertain_decisions": num_uncertain,
            "support_violations": num_violations,
        }
        results.append(result)

        # Log to WandB
        if wandb_enabled:
            wandb.log(
                {
                    "example_id": i,
                    "correct": int(is_correct),
                    "reasoning_steps": num_steps,
                    "decision_count": num_decisions,
                    "uncertain_decisions": num_uncertain,
                    "support_violations": num_violations,
                }
            )

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            acc = correct / (i + 1)
            print(f"Progress: {i + 1}/{len(examples)} | Accuracy: {acc:.3f}")

    # Compute final metrics
    accuracy = correct / len(examples) if examples else 0.0
    avg_steps = total_steps / len(examples) if examples else 0.0
    avg_decisions = total_decisions / len(examples) if examples else 0.0
    avg_uncertain = total_uncertain / len(examples) if examples else 0.0
    violation_rate = total_violations / len(examples) if examples else 0.0

    metrics = {
        "accuracy": accuracy,
        "reasoning_steps": avg_steps,
        "decision_count": avg_decisions,
        "uncertain_decisions": avg_uncertain,
        "support_violations": violation_rate,
        "num_samples": len(examples),
        "num_correct": correct,
    }

    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Log summary to WandB
    if wandb_enabled:
        for key, value in metrics.items():
            wandb.summary[key] = value
        wandb.finish()

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "metrics": metrics,
                "predictions": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")

    # Validation for sanity/pilot modes
    if cfg.mode == "sanity":
        validate_sanity(metrics, results)
    elif cfg.mode == "pilot":
        validate_pilot(metrics, results)

    return metrics


def validate_sanity(metrics: Dict, results: List[Dict]):
    """Validate sanity mode execution."""
    num_samples = metrics["num_samples"]

    # Check minimum samples processed
    if num_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'outputs_valid': 0, 'outputs_unique': 0})}"
        )
        sys.exit(1)

    # Check all outputs valid
    outputs_valid = sum(1 for r in results if r["predicted_answer"])
    outputs_unique = len(
        set(r["predicted_answer"] for r in results if r["predicted_answer"])
    )

    if outputs_valid < num_samples:
        print(f"SANITY_VALIDATION: FAIL reason=invalid_outputs")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'outputs_valid': outputs_valid, 'outputs_unique': outputs_unique})}"
        )
        sys.exit(1)

    # Check metrics are finite
    for key, value in metrics.items():
        if isinstance(value, float) and (
            value != value or value == float("inf") or value == float("-inf")
        ):
            print(f"SANITY_VALIDATION: FAIL reason=non_finite_metrics")
            print(
                f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'outputs_valid': outputs_valid, 'outputs_unique': outputs_unique})}"
            )
            sys.exit(1)

    print("SANITY_VALIDATION: PASS")
    print(
        f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'outputs_valid': outputs_valid, 'outputs_unique': outputs_unique})}"
    )


def validate_pilot(metrics: Dict, results: List[Dict]):
    """Validate pilot mode execution."""
    num_samples = metrics["num_samples"]
    accuracy = metrics["accuracy"]

    # Check minimum samples processed
    if num_samples < 50:
        print(f"PILOT_VALIDATION: FAIL reason=insufficient_samples")
        print(
            f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'outputs_unique': 0})}"
        )
        sys.exit(1)

    # Check outputs unique
    outputs_unique = len(
        set(r["predicted_answer"] for r in results if r["predicted_answer"])
    )

    if outputs_unique < 2:
        print(f"PILOT_VALIDATION: FAIL reason=identical_outputs")
        print(
            f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'outputs_unique': outputs_unique})}"
        )
        sys.exit(1)

    # Check primary metric is finite and logged
    if accuracy != accuracy or accuracy == float("inf") or accuracy == float("-inf"):
        print(f"PILOT_VALIDATION: FAIL reason=non_finite_metrics")
        print(
            f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'outputs_unique': outputs_unique})}"
        )
        sys.exit(1)

    print("PILOT_VALIDATION: PASS")
    print(
        f"PILOT_VALIDATION_SUMMARY: {json.dumps({'samples': num_samples, 'primary_metric': 'accuracy', 'primary_metric_value': accuracy, 'outputs_unique': outputs_unique})}"
    )


if __name__ == "__main__":
    print("This script should be called from main.py with Hydra configuration")
    sys.exit(1)
