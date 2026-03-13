"""Inference script for CCR-CoT and baseline methods."""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import wandb
from omegaconf import DictConfig, OmegaConf

from src.preprocess import load_gsm8k, normalize_number, check_answer_match
from src.model import LLMModel, build_candidate_prompt, build_verification_prompt


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run inference for a single run configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary of metrics and results
    """
    print(f"Starting inference for run: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name}")

    # Initialize WandB if enabled
    wandb_enabled = cfg.wandb.mode == "online"
    if wandb_enabled:
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: Mode check may fail if mode is "sanity_check" instead of "sanity"
        # [CAUSE]: Inconsistent mode naming
        # [FIX]: Check for both "sanity" and "sanity_check" mode values
        #
        # [OLD CODE]:
        # if cfg.mode == "sanity":
        #     project = f"{project}-sanity"
        # elif cfg.mode == "pilot":
        #     project = f"{project}-pilot"
        #
        # [NEW CODE]:
        # Adjust project name based on mode
        project = cfg.wandb.project
        if cfg.mode in ["sanity", "sanity_check"]:
            project = f"{project}-sanity"
        elif cfg.mode == "pilot":
            project = f"{project}-pilot"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB initialized: {wandb.run.url}")

    # Load dataset
    print("Loading dataset...")
    dataset_cfg = cfg.run.dataset
    examples = load_gsm8k(
        split=dataset_cfg.split,
        subset_size=dataset_cfg.subset_size,
        cache_dir=dataset_cfg.cache_dir,
    )

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Mode check may fail if mode is "sanity_check" instead of "sanity"
    # [CAUSE]: Inconsistent mode naming between runner and code
    # [FIX]: Check for both "sanity" and "sanity_check" mode values
    #
    # [OLD CODE]:
    # if cfg.mode == "sanity":
    #     examples = examples[:10]  # 10 samples for sanity
    #     print(f"Sanity mode: using {len(examples)} samples")
    # elif cfg.mode == "pilot":
    #     pilot_size = max(50, int(len(examples) * 0.2))  # 20% or at least 50
    #     examples = examples[:pilot_size]
    #     print(f"Pilot mode: using {len(examples)} samples")
    #
    # [NEW CODE]:
    # Apply mode-specific subset
    if cfg.mode in ["sanity", "sanity_check"]:
        examples = examples[:10]  # 10 samples for sanity
        print(f"Sanity mode: using {len(examples)} samples")
    elif cfg.mode == "pilot":
        pilot_size = max(50, int(len(examples) * 0.2))  # 20% or at least 50
        examples = examples[:pilot_size]
        print(f"Pilot mode: using {len(examples)} samples")

    # Initialize model
    print("Initializing model...")
    # [VALIDATOR FIX - Attempt 2]
    # [PROBLEM]: OpenAI API returns JSON parsing errors intermittently
    # [CAUSE]: OmegaConf values (from Hydra) are not pure Python types and may not serialize correctly to JSON
    # [FIX]: Explicitly convert OmegaConf values to native Python types before passing to API
    #
    # [OLD CODE]:
    # model = LLMModel(
    #     provider=cfg.run.model.provider,
    #     model_name=cfg.run.model.name,
    #     api_key_env=cfg.run.model.api_key_env,
    #     temperature=cfg.run.model.temperature,
    #     max_tokens=cfg.run.model.max_tokens,
    # )
    #
    # [NEW CODE]:
    model = LLMModel(
        provider=str(cfg.run.model.provider),
        model_name=str(cfg.run.model.name),
        api_key_env=str(cfg.run.model.api_key_env),
        temperature=float(cfg.run.model.temperature),
        max_tokens=int(cfg.run.model.max_tokens),
    )

    # Run inference based on method
    method_name = cfg.run.method.name

    if method_name == "ccr-cot":
        results = run_ccr_cot(cfg, model, examples)
    elif method_name == "self-consistency":
        results = run_self_consistency(cfg, model, examples)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Compute metrics
    metrics = compute_metrics(results)
    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f}")
    print(f"Correct: {metrics['num_correct']}/{metrics['num_total']}")

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Results not being saved to the correct directory
    # [CAUSE]: Hydra changes working directory, making relative paths point to Hydra outputs dir
    # [FIX]: Convert results_dir to absolute path before using it
    #
    # [OLD CODE]:
    # results_dir = Path(cfg.results_dir) / cfg.run.run_id
    # results_dir.mkdir(parents=True, exist_ok=True)
    #
    # [NEW CODE]:
    # Save results with absolute path to avoid Hydra working directory issues
    from hydra.utils import get_original_cwd

    results_base = Path(get_original_cwd()) / cfg.results_dir
    results_dir = results_base / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(results_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {results_dir}")

    # Log to WandB
    if wandb_enabled:
        wandb.log(metrics)
        wandb.summary.update(metrics)
        wandb.finish()

    return metrics


def run_ccr_cot(cfg: DictConfig, model: LLMModel, examples: List[Dict]) -> List[Dict]:
    """
    Run CCR-CoT method: generate candidates, run verification probes, re-rank.

    Args:
        cfg: Configuration
        model: LLM model
        examples: List of questions with ground truth

    Returns:
        List of results per example
    """
    results = []
    method_cfg = cfg.run.method

    num_candidates = method_cfg.num_candidates
    num_probes = method_cfg.num_probes
    probe_types = method_cfg.probe_types
    score_weights = method_cfg.score_weights
    fallback_threshold = method_cfg.fallback_threshold

    for i, example in enumerate(examples):
        print(f"\nProcessing {i + 1}/{len(examples)}: {example['question'][:60]}...")

        # Step 1: Generate K candidates
        prompt = build_candidate_prompt(example["question"], method_cfg.name)
        candidates = model.generate(prompt, n=num_candidates)

        # Step 2: For each candidate, run M verification probes
        candidate_scores = []
        for cand_idx, candidate in enumerate(candidates):
            # Extract answer from candidate
            cand_answer = extract_final_answer(candidate)

            # Run verification probes
            probe_results = []
            for probe_idx in range(num_probes):
                probe_type = probe_types[probe_idx % len(probe_types)]
                probe_prompt = build_verification_prompt(
                    example["question"],
                    candidate,
                    probe_type,
                    max_words=method_cfg.probe_max_tokens,
                )

                probe_response = model.generate(probe_prompt, max_tokens=100, n=1)[0]

                # Parse verdict and answer
                verdict = parse_verdict(probe_response)
                probe_answer = extract_final_answer(probe_response)

                probe_results.append(
                    {
                        "type": probe_type,
                        "verdict": verdict,
                        "answer": probe_answer,
                        "response": probe_response,
                    }
                )

            # Compute consistency score
            score = compute_consistency_score(
                candidate, cand_answer, probe_results, score_weights
            )

            candidate_scores.append(
                {
                    "candidate": candidate,
                    "answer": cand_answer,
                    "probes": probe_results,
                    "score": score,
                }
            )

        # Step 3: Select best candidate or fallback to majority vote
        best_candidate = max(candidate_scores, key=lambda x: x["score"])

        if best_candidate["score"] >= fallback_threshold:
            final_answer = best_candidate["answer"]
            selection_method = "ccr_score"
        else:
            # Fallback to majority vote
            all_answers = [c["answer"] for c in candidate_scores]
            final_answer = majority_vote(all_answers)
            selection_method = "majority_vote_fallback"

        # Check correctness
        is_correct = check_answer_match(final_answer, example["answer"])

        results.append(
            {
                "question": example["question"],
                "ground_truth": example["answer"],
                "prediction": final_answer,
                "correct": is_correct,
                "selection_method": selection_method,
                "candidates": candidate_scores,
                "best_score": best_candidate["score"],
            }
        )

        print(
            f"  Predicted: {final_answer}, Ground truth: {example['answer']}, "
            f"Correct: {is_correct}, Score: {best_candidate['score']:.3f}"
        )

    return results


def run_self_consistency(
    cfg: DictConfig, model: LLMModel, examples: List[Dict]
) -> List[Dict]:
    """
    Run Self-Consistency baseline: generate candidates, majority vote.

    Args:
        cfg: Configuration
        model: LLM model
        examples: List of questions with ground truth

    Returns:
        List of results per example
    """
    results = []
    method_cfg = cfg.run.method

    num_candidates = method_cfg.num_candidates

    for i, example in enumerate(examples):
        print(f"\nProcessing {i + 1}/{len(examples)}: {example['question'][:60]}...")

        # Generate K candidates
        prompt = build_candidate_prompt(example["question"], method_cfg.name)
        candidates = model.generate(prompt, n=num_candidates)

        # Extract answers from all candidates
        candidate_answers = []
        for candidate in candidates:
            answer = extract_final_answer(candidate)
            candidate_answers.append(answer)

        # Majority vote
        final_answer = majority_vote(candidate_answers)

        # Check correctness
        is_correct = check_answer_match(final_answer, example["answer"])

        results.append(
            {
                "question": example["question"],
                "ground_truth": example["answer"],
                "prediction": final_answer,
                "correct": is_correct,
                "selection_method": "majority_vote",
                "candidates": [{"answer": ans} for ans in candidate_answers],
            }
        )

        print(
            f"  Predicted: {final_answer}, Ground truth: {example['answer']}, "
            f"Correct: {is_correct}"
        )

    return results


def extract_final_answer(text: str) -> float:
    """
    Extract the final numeric answer from a solution text.
    Looks for explicit markers like "COMMIT:", "ANSWER:", or the last number.

    Args:
        text: Solution text

    Returns:
        Extracted numeric answer
    """
    if not text:
        return None

    # Look for explicit answer markers
    patterns = [
        r"COMMIT:\s*([0-9,.-]+)",
        r"ANSWER:\s*([0-9,.-]+)",
        r"final answer.*?([0-9,.-]+)",
        r"answer is.*?([0-9,.-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return normalize_number(match.group(1))

    # Fallback: extract the last number in the text
    return normalize_number(text)


def parse_verdict(text: str) -> str:
    """
    Parse PASS/FAIL verdict from verification probe response.

    Args:
        text: Probe response text

    Returns:
        "PASS" or "FAIL"
    """
    text_upper = text.upper()

    # Look for explicit VERDICT
    match = re.search(r"VERDICT:\s*(PASS|FAIL)", text_upper)
    if match:
        return match.group(1)

    # Fallback: check for keywords
    if "PASS" in text_upper:
        return "PASS"
    elif "FAIL" in text_upper or "INCORRECT" in text_upper or "WRONG" in text_upper:
        return "FAIL"

    # Default: assume PASS if uncertain
    return "PASS"


def compute_consistency_score(
    candidate: str, cand_answer: float, probe_results: List[Dict], score_weights: Dict
) -> float:
    """
    Compute consistency score from verification probes.

    Args:
        candidate: Candidate solution text
        cand_answer: Candidate's final answer
        probe_results: List of probe results
        score_weights: Weights for score components

    Returns:
        Consistency score (0 to 1)
    """
    if not probe_results:
        return 0.0

    # Pass rate: fraction of PASS verdicts
    pass_count = sum(1 for p in probe_results if p["verdict"] == "PASS")
    pass_rate = pass_count / len(probe_results)

    # Answer stability: fraction of probes that agree with candidate answer
    if cand_answer is not None:
        agree_count = sum(
            1
            for p in probe_results
            if p["answer"] is not None and abs(p["answer"] - cand_answer) < 1e-4
        )
        answer_stability = agree_count / len(probe_results)
    else:
        answer_stability = 0.0

    # Brevity penalty: normalized by reasonable length (penalize overly long)
    # Typical CoT solutions are 100-300 words; penalize beyond 400
    word_count = len(candidate.split())
    brevity_penalty = max(0, 1 - (word_count - 400) / 400) if word_count > 400 else 1.0

    # Combine scores
    score = (
        score_weights.pass_rate * pass_rate
        + score_weights.answer_stability * answer_stability
        + score_weights.brevity_penalty * brevity_penalty
    )

    return score


def majority_vote(answers: List[float]) -> float:
    """
    Select answer by majority vote. If tie, pick the first.

    Args:
        answers: List of numeric answers

    Returns:
        Most common answer
    """
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None

    # Round to avoid floating point issues
    rounded = [round(a, 2) for a in valid_answers]

    # Count occurrences
    counter = Counter(rounded)
    most_common = counter.most_common(1)[0][0]

    return most_common


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute evaluation metrics from results.

    Args:
        results: List of inference results

    Returns:
        Dictionary of metrics
    """
    num_total = len(results)
    num_correct = sum(1 for r in results if r["correct"])
    accuracy = num_correct / num_total if num_total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
    }

    # Add method-specific metrics if available
    if "best_score" in results[0]:
        avg_score = sum(r["best_score"] for r in results) / num_total
        metrics["avg_consistency_score"] = avg_score

    return metrics


if __name__ == "__main__":
    print("This script should be run via src.main, not directly.")
    sys.exit(1)
