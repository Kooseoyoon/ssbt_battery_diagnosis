"""Synthetic plumbing demo. No real-data result or adaptive testing claim."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .core import pack_lifetimes, sorted_assignment, utility

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/demo.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    n, m, s = config["n_cells"], config["pack_size"], config["n_scenarios"]
    if s < 1:
        raise ValueError("At least one scenario is required.")

    # All three arrays below are synthetic. They are not fitted to battery data.
    true_rul = rng.uniform(80, 400, size=n)
    sigma = rng.uniform(10, 65, size=n)
    predicted_mean = np.maximum(1, true_rul + rng.normal(0, sigma))
    # Clipping is a demo convenience, not a validated RUL predictive model.
    scenarios = np.maximum(0, rng.normal(predicted_mean, sigma, size=(s, n)))

    # Decision uses predictions only; hidden truth is used only in evaluation.
    chosen = sorted_assignment(predicted_mean, m)
    oracle = sorted_assignment(true_rul, m)
    actual = float(utility(true_rul, chosen))
    oracle_value = float(utility(true_rul, oracle))
    scenario_values = utility(scenarios, chosen)
    perfect_information = np.array([
        utility(row, sorted_assignment(row, m)) for row in scenarios
    ])

    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    report = {
        "data_kind": "SYNTHETIC_DEMO_ONLY",
        "notes": [
            "No real battery data or learned prediction model is used.",
            "No sequential diagnostic policy or diagnostic cost is implemented yet.",
            "The chosen assignment is fixed across scenarios.",
            "Sorting predicted means is a baseline, not an expected-utility optimality claim.",
        ],
        "config": config,
        "actual_utility_pack_cycles": actual,
        "oracle_utility_pack_cycles": oracle_value,
        "relative_regret": (oracle_value - actual) / oracle_value,
        "predicted_expected_utility_fixed_assignment": float(scenario_values.mean()),
        "scenario_perfect_information_upper_bound": float(perfect_information.mean()),
        "assignment_cell_indices": chosen.tolist(),
    }
    report_path = results / "demo_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez(results / "demo_synthetic.npz", true_rul=true_rul,
             predicted_mean=predicted_mean, sigma=sigma, scenarios=scenarios,
             assignment=chosen)

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    x = np.arange(chosen.shape[0])
    ax.bar(x - .2, pack_lifetimes(true_rul, chosen), .4, label="Predicted-mean sorting")
    ax.bar(x + .2, pack_lifetimes(true_rul, oracle), .4, label="Oracle (hidden truth)")
    ax.set(xlabel="Pack index (each method's own order)", ylabel="Actual pack RUL (cycles)",
           title="SYNTHETIC DEMO: pack-lifetime scoring")
    ax.legend()
    fig.savefig(figures / "demo_pack_lifetimes.png", dpi=180)
    plt.close(fig)
    print("합성 데이터 예제 완료 — 실제 연구 결과가 아닙니다.")
    print(f"셀 {n}개 / 팩당 {m}개 / 시나리오 {s}개")
    print(f"실제 수명으로 평가한 구성 성능: {actual:.1f} pack-cycles")
    print(f"정답을 아는 최적 구성의 성능: {oracle_value:.1f} pack-cycles")
    print(f"상대 regret: {report['relative_regret']:.2%}")
    print(f"결과: {report_path.relative_to(ROOT).as_posix()}")
    print(f"그림: {(figures / 'demo_pack_lifetimes.png').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
