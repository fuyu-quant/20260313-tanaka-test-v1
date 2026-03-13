"""Main orchestrator for inference experiments."""

import sys
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Starting experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Validate required configuration
    if not hasattr(cfg, "run") or not hasattr(cfg.run, "run_id"):
        print("ERROR: Missing required configuration. Please specify run=<run_id>")
        sys.exit(1)

    # Run inference
    try:
        metrics = run_inference(cfg)

        # Perform validation based on mode
        if cfg.mode == "sanity":
            validate_sanity(cfg, metrics)
        elif cfg.mode == "pilot":
            validate_pilot(cfg, metrics)

        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def validate_sanity(cfg: DictConfig, metrics: dict) -> None:
    """
    Validate sanity mode run.

    Args:
        cfg: Configuration
        metrics: Computed metrics
    """
    print("\n" + "=" * 80)
    print("SANITY MODE VALIDATION")
    print("=" * 80)

    # Check if we processed enough samples
    num_samples = metrics.get("num_total", 0)

    # Check if metrics are valid
    accuracy = metrics.get("accuracy", 0)

    # Validation criteria for inference task
    validation_passed = True
    failure_reason = None

    # Must process at least 5 samples
    if num_samples < 5:
        validation_passed = False
        failure_reason = "insufficient_samples"

    # Metrics must be finite (no NaN/inf)
    elif not (0 <= accuracy <= 1):
        validation_passed = False
        failure_reason = "invalid_metrics"

    # For sanity, accuracy can be 0 (it's a quick test), but should have run

    # Print summary
    summary = {
        "samples": num_samples,
        "accuracy": accuracy,
        "num_correct": metrics.get("num_correct", 0),
    }

    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    if validation_passed:
        print("SANITY_VALIDATION: PASS")
    else:
        print(f"SANITY_VALIDATION: FAIL reason={failure_reason}")
        sys.exit(1)


def validate_pilot(cfg: DictConfig, metrics: dict) -> None:
    """
    Validate pilot mode run.

    Args:
        cfg: Configuration
        metrics: Computed metrics
    """
    print("\n" + "=" * 80)
    print("PILOT MODE VALIDATION")
    print("=" * 80)

    # Check if we processed enough samples
    num_samples = metrics.get("num_total", 0)

    # Get primary metric
    accuracy = metrics.get("accuracy", 0)

    # Validation criteria for inference task
    validation_passed = True
    failure_reason = None

    # Must process at least 50 samples
    if num_samples < 50:
        validation_passed = False
        failure_reason = "insufficient_samples"

    # Metrics must be finite and valid
    elif not (0 <= accuracy <= 1):
        validation_passed = False
        failure_reason = "invalid_metrics"

    # Primary metric should be non-zero (sanity check that method is working)
    # Note: accuracy could legitimately be 0 for a bad method, so we don't fail on this

    # Print summary
    summary = {
        "samples": num_samples,
        "primary_metric": "accuracy",
        "primary_metric_value": accuracy,
        "num_correct": metrics.get("num_correct", 0),
    }

    if "avg_consistency_score" in metrics:
        summary["avg_consistency_score"] = metrics["avg_consistency_score"]

    print(f"PILOT_VALIDATION_SUMMARY: {json.dumps(summary)}")

    if validation_passed:
        print("PILOT_VALIDATION: PASS")
    else:
        print(f"PILOT_VALIDATION: FAIL reason={failure_reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
