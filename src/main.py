"""Main orchestrator for inference experiments."""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for running inference experiments.

    This orchestrator:
    1. Loads Hydra configuration
    2. Applies mode-specific overrides
    3. Runs inference
    4. Validates results based on mode
    """
    print("=" * 80)
    print("AIRAS Prompt-Tuning Experiment")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Validate configuration
    if not hasattr(cfg, "run") or not hasattr(cfg.run, "run_id"):
        print("Error: Missing run configuration. Use: run={run_id}")
        sys.exit(1)

    # Run inference
    try:
        metrics = run_inference(cfg)
        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)
        return metrics
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
