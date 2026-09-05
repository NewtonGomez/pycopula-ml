#!/usr/bin/env python3
"""Monte Carlo validation of Frank-copula theta estimation.

Purpose
-------
It is a statistical validation experiment intended to
live under ``scripts/validation``.

Workflow
--------
1. Generate pseudo-observations (u, v) from a Frank copula with a known
   ``theta_true`` using statsmodels as an independent generator.
2. Estimate theta by maximizing the log-likelihood implemented in
   ``pycopula_ml``.
3. Repeat the experiment several times.
4. Report bias, standard deviation, RMSE and individual estimates.

The current version uses positive theta because statsmodels 0.14.x sampling for
Frank copulas relies on a representation that does not support negative theta.

Example
-------
python3 scripts/validation/validate_frank_theta.py \
    --theta 5 \
    --samples 1000 \
    --trials 30 \
    --seed 42 \
    --output frank_theta_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from statsmodels.distributions.copula.api import FrankCopula as SMFrankCopula

from pycopula_ml.copulas.frank import FrankCopula


@dataclass
class TrialResult:
    """Result of one Monte Carlo estimation trial."""

    trial: int
    theta_true: float
    theta_hat: float
    error: float
    abs_error: float
    relative_error: float
    log_likelihood: float
    optimizer_success: bool


def estimate_theta(
    u: np.ndarray,
    v: np.ndarray,
    lower: float = 1e-4,
    upper: float = 50.0,
) -> tuple[float, float, bool]:
    """Estimate positive Frank theta by maximum likelihood."""

    def objective(theta: float) -> float:
        copula = FrankCopula(theta=float(theta))
        return -float(copula.log_likelihood(u, v))

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10},
    )

    theta_hat = float(result.x)
    log_likelihood = -float(result.fun)

    return theta_hat, log_likelihood, bool(result.success)


def run_experiment(
    theta_true: float,
    samples: int,
    trials: int,
    seed: int,
    upper_bound: float,
) -> list[TrialResult]:
    """Run repeated simulations with known theta."""

    if theta_true <= 0:
        raise ValueError(
            "This validation script currently requires theta > 0 because "
            "statsmodels 0.14.x cannot sample negative-theta Frank copulas."
        )

    if samples < 2:
        raise ValueError("samples must be >= 2")

    if trials < 1:
        raise ValueError("trials must be >= 1")

    if upper_bound <= theta_true:
        raise ValueError(
            "upper-bound must be larger than theta_true so the optimizer "
            "does not truncate the estimate."
        )

    master_rng = np.random.default_rng(seed)
    results: list[TrialResult] = []

    for trial in range(1, trials + 1):
        trial_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        rng = np.random.default_rng(trial_seed)

        reference_copula = SMFrankCopula(theta=theta_true)
        uv = np.asarray(
            reference_copula.rvs(samples, random_state=rng),
            dtype=float,
        )

        u = uv[:, 0]
        v = uv[:, 1]

        theta_hat, log_likelihood, success = estimate_theta(
            u,
            v,
            upper=upper_bound,
        )

        error = theta_hat - theta_true
        abs_error = abs(error)
        relative_error = abs_error / abs(theta_true)

        results.append(
            TrialResult(
                trial=trial,
                theta_true=theta_true,
                theta_hat=theta_hat,
                error=error,
                abs_error=abs_error,
                relative_error=relative_error,
                log_likelihood=log_likelihood,
                optimizer_success=success,
            )
        )

    return results


def print_summary(results: list[TrialResult]) -> None:
    """Print the Monte Carlo summary."""

    theta_true = results[0].theta_true
    estimates = np.array([r.theta_hat for r in results], dtype=float)
    errors = estimates - theta_true

    mean_hat = float(np.mean(estimates))
    std_hat = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    min_hat = float(np.min(estimates))
    max_hat = float(np.max(estimates))
    success_rate = float(
        np.mean([r.optimizer_success for r in results])
    )

    print("\nFrank theta estimation validation")
    print("-" * 48)
    print(f"True theta:          {theta_true:.8f}")
    print(f"Trials:              {len(results)}")
    print(f"Mean(theta_hat):     {mean_hat:.8f}")
    print(f"Std(theta_hat):      {std_hat:.8f}")
    print(f"Bias:                {bias:.8f}")
    print(f"MAE:                 {mae:.8f}")
    print(f"RMSE:                {rmse:.8f}")
    print(f"Minimum theta_hat:   {min_hat:.8f}")
    print(f"Maximum theta_hat:   {max_hat:.8f}")
    print(f"Optimizer success:   {success_rate:.2%}")


def save_csv(results: list[TrialResult], output: Path) -> None:
    """Save trial-level results to CSV."""

    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial",
        "theta_true",
        "theta_hat",
        "error",
        "abs_error",
        "relative_error",
        "log_likelihood",
        "optimizer_success",
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "trial": result.trial,
                    "theta_true": result.theta_true,
                    "theta_hat": result.theta_hat,
                    "error": result.error,
                    "abs_error": result.abs_error,
                    "relative_error": result.relative_error,
                    "log_likelihood": result.log_likelihood,
                    "optimizer_success": result.optimizer_success,
                }
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate Frank-copula theta estimation by Monte Carlo.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=5.0,
        help="Known positive Frank theta used to generate data (default: 5).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Observations per trial (default: 1000).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Monte Carlo trials (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42).",
    )
    parser.add_argument(
        "--upper-bound",
        type=float,
        default=50.0,
        help="Upper bound for theta optimization (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for trial-level results.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the validation experiment."""

    args = parse_args()

    results = run_experiment(
        theta_true=args.theta,
        samples=args.samples,
        trials=args.trials,
        seed=args.seed,
        upper_bound=args.upper_bound,
    )

    print_summary(results)

    if args.output is not None:
        save_csv(results, args.output)
        print(f"\nCSV saved to: {args.output}")


if __name__ == "__main__":
    main()
