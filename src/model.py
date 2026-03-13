"""LLM model interface for generating Chain-of-Thought reasoning."""

import os
import time
from typing import List, Dict, Any, Optional
import openai


class LLMModel:
    """Wrapper for LLM API calls with retry logic."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key_env: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ):
        """
        Initialize LLM model.

        Args:
            provider: API provider (e.g., "openai")
            model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key_env: Environment variable name for API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")

        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        n: int = 1,
    ) -> List[str]:
        """
        Generate completions from the model.

        Args:
            prompt: Input prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            n: Number of completions to generate

        Returns:
            List of generated text completions
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        # Retry logic for API failures
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )

                # Extract completions
                completions = [choice.message.content for choice in response.choices]
                return completions

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise

        return []  # Should not reach here


def build_candidate_prompt(question: str, method_name: str) -> str:
    """
    Build prompt for generating a candidate solution.

    Args:
        question: Math problem question
        method_name: Method name (for different prompt templates)

    Returns:
        Formatted prompt string
    """
    if method_name == "ccr-cot":
        # Structured template for CCR-CoT
        prompt = f"""Solve the following math problem step by step.

Problem: {question}

Please structure your solution as follows:

PLAN:
[Write a minimal bullet-point plan of your approach in 1-3 bullets]

EXECUTE:
[Work through the problem concisely, showing your calculations]

COMMIT:
[State your final answer as a single number]

Your response:"""

    elif method_name == "self-consistency":
        # Standard CoT prompt for self-consistency baseline
        prompt = f"""Solve the following math problem step by step.

Problem: {question}

Think step by step and show your work. End with your final answer.

Your response:"""

    else:
        raise ValueError(f"Unknown method: {method_name}")

    return prompt


def build_verification_prompt(
    question: str, candidate_solution: str, probe_type: str, max_words: int = 80
) -> str:
    """
    Build prompt for counterfactual verification probe.

    Args:
        question: Original math problem
        candidate_solution: Candidate solution to verify
        probe_type: Type of verification probe
        max_words: Maximum words for probe response

    Returns:
        Formatted verification prompt
    """
    base_instruction = (
        f"Original problem: {question}\n\nProposed solution:\n{candidate_solution}\n\n"
    )

    if probe_type == "alternate_representation":
        instruction = (
            f"{base_instruction}"
            f"Task: Express the same problem and solution using a different representation "
            f"(e.g., if the solution used words, use equations; if it used equations, use words). "
            f"Then verify if the answer is correct.\n\n"
            f"In at most {max_words} words, provide:\n"
            f"VERDICT: [PASS or FAIL]\n"
            f"ANSWER: [single number]"
        )

    elif probe_type == "reverse_check":
        instruction = (
            f"{base_instruction}"
            f"Task: Plug the proposed answer back into the problem or recompute using a different order. "
            f"Verify if it satisfies all conditions.\n\n"
            f"In at most {max_words} words, provide:\n"
            f"VERDICT: [PASS or FAIL]\n"
            f"ANSWER: [single number]"
        )

    elif probe_type == "sanity_check":
        instruction = (
            f"{base_instruction}"
            f"Task: Check if the answer is reasonable in terms of units, dimensions, or bounds. "
            f"Does the magnitude make sense given the problem?\n\n"
            f"In at most {max_words} words, provide:\n"
            f"VERDICT: [PASS or FAIL]\n"
            f"ANSWER: [single number]"
        )

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    return instruction
