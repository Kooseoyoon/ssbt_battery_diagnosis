"""Constitutive-equation PINN pilot for cycle-1/6 capacity inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from .physics_experiment import (
    ANCHOR_CYCLE,
    _input_path,
    _validate_cell_history,
    early_features,
    predicted_rul,
    query_rows,
    rank_predicted_rul,
    trajectory_samples,
    tune_model,
)
from .shared_folds import validate_folds
from .value_metrics import current_only_assignment, total_value

ROOT = Path(__file__).resolve().parents[1]
EQUATIONS = ("sqrt_linear", "exponential_linear")
LAMBDA_FIELD_NAMES = {
    "sqrt_linear": ("k_s_Ah_per_sqrt_RPT", "k_l_Ah_per_RPT"),
    "exponential_linear": ("k_exp_Ah", "tau_RPT", "k_l_Ah_per_RPT"),
}


def constitutive_rate(equation, x, parameters, epsilon):
    """Positive degradation rate -dQ/dx for the two selected ODEs."""
    if equation == "sqrt_linear":
        k_s, k_l = parameters[..., 0], parameters[..., 1]
        return k_s / (2.0 * torch.sqrt(x + epsilon)) + k_l
    if equation == "exponential_linear":
        k_exp, tau, k_l = parameters[..., 0], parameters[..., 1], parameters[..., 2]
        return k_exp / tau * torch.exp(-x / tau) + k_l
    raise ValueError(f"Unknown constitutive equation: {equation}")


class _ConstitutiveNetwork(nn.Module):
    def __init__(self, hidden_units, n_parameters):
        super().__init__()
        self.capacity = nn.Sequential(
            nn.Linear(4, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),
        )
        self.coefficients = (
            nn.Sequential(
                nn.Linear(3, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, n_parameters),
            )
            if n_parameters
            else None
        )


def _positive_coefficients(model, equation, scaled_base, config):
    raw = model.coefficients(scaled_base)
    if equation == "sqrt_linear":
        scales = torch.as_tensor(
            [config["sqrt_k_s_scale"], config["linear_k_scale"]],
            dtype=raw.dtype,
            device=raw.device,
        )
        return F.softplus(raw) * scales
    scales = torch.as_tensor(
        [
            config["exponential_k_scale"],
            config["exponential_tau_scale"],
            config["linear_k_scale"],
        ],
        dtype=raw.dtype,
        device=raw.device,
    )
    positive = F.softplus(raw) * scales
    return torch.cat(
        (
            positive[:, :1],
            positive[:, 1:2] + float(config["minimum_tau_RPT"]),
            positive[:, 2:],
        ),
        dim=1,
    )


def fit_constitutive_model(
    equation,
    physics_weight,
    capacities,
    cycles,
    eol,
    train_cells,
    x0,
    config,
    seed,
):
    """Fit one shared trajectory net, with an optional ODE-residual coefficient head."""
    train_cells = np.asarray(train_cells, dtype=int)
    raw_x, raw_y, owners = trajectory_samples(
        capacities, cycles, eol, train_cells, config
    )
    x_scaler = StandardScaler().fit(x0[train_cells])
    scaled_cells = x_scaler.transform(x0[train_cells])
    local_index = {int(cell): index for index, cell in enumerate(train_cells)}
    owner_positions = np.asarray([local_index[int(cell)] for cell in owners], dtype=int)
    scaled_rows = scaled_cells[owner_positions]
    c6_column = int(np.flatnonzero(cycles == ANCHOR_CYCLE)[0])
    c6_by_owner = capacities[owners, c6_column]
    capacity_scale = max(
        float(np.std(raw_y - c6_by_owner)),
        float(config["minimum_trajectory_scale_Ah"]),
    )
    n_parameters = len(LAMBDA_FIELD_NAMES[equation]) if equation else 0
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _ConstitutiveNetwork(int(config["hidden_units"]), n_parameters)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    base_data = torch.as_tensor(scaled_rows, dtype=torch.float32)
    u_data = torch.as_tensor(raw_x[:, 3], dtype=torch.float32)
    c6_data = torch.as_tensor(c6_by_owner, dtype=torch.float32)
    y_data = torch.as_tensor(raw_y, dtype=torch.float32)
    data_scale = float(config["data_huber_scale_Ah"])
    max_x = float(config["maximum_predicted_rul_cycles"]) / 5.0
    eps = float(config["sqrt_epsilon_RPT"])

    if equation:
        collocation_x, collocation_owners = [], []
        for position, cell in enumerate(train_cells):
            end = _validate_cell_history(
                capacities[cell], cycles, eol[cell], config["eol_capacity_Ah"]
            )
            maximum_cell_x = (float(cycles[end]) - ANCHOR_CYCLE) / 5.0
            collocation_x.extend(
                np.linspace(0.0, maximum_cell_x, int(config["collocation_points_per_cell"]))
            )
            collocation_owners.extend([position] * int(config["collocation_points_per_cell"]))
        collocation_x = np.asarray(collocation_x, dtype=np.float32)
        collocation_owners = np.asarray(collocation_owners, dtype=int)
        x_physics = torch.as_tensor(collocation_x, dtype=torch.float32).requires_grad_(True)
        base_physics = torch.as_tensor(scaled_cells[collocation_owners], dtype=torch.float32)
        c6_physics = torch.as_tensor(
            capacities[train_cells[collocation_owners], c6_column], dtype=torch.float32
        )
        physics_scale = float(config["physics_residual_scale_Ah_per_RPT"])
    else:
        x_physics = base_physics = c6_physics = None

    def anchored_capacity(scaled_base, u, c6):
        current = model.capacity(torch.cat((scaled_base, u.reshape(-1, 1)), dim=1)).reshape(-1)
        origin = model.capacity(
            torch.cat((scaled_base, torch.zeros_like(u).reshape(-1, 1)), dim=1)
        ).reshape(-1)
        return c6 + capacity_scale * (current - origin)

    last_data_loss = last_physics_loss = float("nan")
    for _ in range(int(config["epochs"])):
        optimizer.zero_grad()
        predicted = anchored_capacity(base_data, u_data, c6_data)
        data_loss = F.smooth_l1_loss((predicted - y_data) / data_scale, torch.zeros_like(y_data))
        loss = data_loss
        physics_loss = torch.zeros((), dtype=torch.float32)
        if equation:
            u_physics = x_physics * (
                5.0 / float(config["maximum_predicted_rul_cycles"])
            )
            predicted_physics = anchored_capacity(base_physics, u_physics, c6_physics)
            derivative_x = torch.autograd.grad(
                predicted_physics.sum(), x_physics, create_graph=True
            )[0]
            parameter_rows = _positive_coefficients(
                model, equation, base_physics, config
            )
            residual = derivative_x + constitutive_rate(
                equation, x_physics, parameter_rows, eps
            )
            physics_loss = torch.mean((residual / physics_scale) ** 2)
            loss = loss + float(physics_weight) * physics_loss
        loss.backward()
        optimizer.step()
        last_data_loss = float(data_loss.detach())
        last_physics_loss = float(physics_loss.detach())

    def predict(base_features, query_cycles):
        base = np.asarray(base_features, dtype=float).reshape(1, -1)
        queried_cycles = np.asarray(query_cycles, dtype=float)
        scaled_base = x_scaler.transform(base)
        repeated_base = np.repeat(scaled_base, len(queried_cycles), axis=0)
        u = (queried_cycles - ANCHOR_CYCLE) / float(config["maximum_predicted_rul_cycles"])
        with torch.no_grad():
            base_tensor = torch.as_tensor(repeated_base, dtype=torch.float32)
            u_tensor = torch.as_tensor(u, dtype=torch.float32)
            c6_tensor = torch.full((len(u),), float(base[0, 1]), dtype=torch.float32)
            forecast = anchored_capacity(base_tensor, u_tensor, c6_tensor).cpu().numpy()
            if model.coefficients is None:
                parameter_values = None
            else:
                parameter_tensor = _positive_coefficients(
                    model,
                    equation,
                    torch.as_tensor(scaled_base, dtype=torch.float32),
                    config,
                )
                parameter_values = parameter_tensor.cpu().numpy()[0]
        return forecast, parameter_values

    return predict, {
        "training_data_loss": last_data_loss,
        "training_physics_residual_loss": last_physics_loss,
        "trained_cell_count": int(len(train_cells)),
        "training_point_count": int(len(raw_y)),
        "capacity_residual_scale_Ah": capacity_scale,
        "physics_residual_scale_Ah_per_RPT": (
            float(config["physics_residual_scale_Ah_per_RPT"]) if equation else None
        ),
    }


def _evaluate_curve(predict, base, cell, capacities, cycles, eol, config):
    start_cycle = int(config["diagnosis_cycles"]) + 1
    maximum_rul = float(config["maximum_predicted_rul_cycles"])
    horizon = start_cycle + 5.0 * np.arange(int(maximum_rul // 5) + 1)
    predicted_horizon, parameters = predict(base, horizon)
    rul, status = predicted_rul(
        predicted_horizon,
        horizon,
        start_cycle,
        config["eol_capacity_Ah"],
        maximum_rul,
    )
    end = _validate_cell_history(
        capacities[cell], cycles, eol[cell], config["eol_capacity_Ah"]
    )
    first_future = int(np.flatnonzero(cycles == start_cycle)[0])
    actual_columns = np.arange(first_future, end + 1)
    actual_curve, _ = predict(base, cycles[actual_columns])
    trajectory_mae = float(
        np.mean(np.abs(actual_curve - capacities[cell, actual_columns]))
    )
    return {
        "predicted_RUL_cycles": rul,
        "predicted_RUL_status": status,
        "capacity_trajectory_MAE_Ah_evaluation_only": trajectory_mae,
        "actual_RUL_cycles_evaluation_only": float(eol[cell] - start_cycle),
        "actual_T_eol_evaluation_only": int(eol[cell]),
        "predicted_capacity_at_cycle_11_Ah": float(predicted_horizon[0]),
    }, parameters


def _candidate_id(equation, physics_weight):
    if equation == "neural_network":
        return equation
    return f"pinn_{equation}_lambda_{float(physics_weight):g}"


def run_experiment(config):
    data_dir = _input_path(config["data_dir"])
    capacity_path, eol_path = data_dir / "C_full.npz", data_dir / "T_eol.npy"
    fold_path = _input_path(config["folds_path"])
    with np.load(capacity_path, allow_pickle=False) as archive:
        capacities = archive["C_full"].copy()
        ids = archive["cell_ids"].copy()
        cycles = archive["cycles"].copy()
    eol = np.load(eol_path, allow_pickle=False)
    folds = validate_folds(pd.read_csv(fold_path), ids)
    fold_by_id = dict(zip(folds["cell_id"], folds["fold"]))
    fold_ids = np.asarray([fold_by_id[cell_id] for cell_id in ids], dtype=int)
    if capacities.shape[0] != 86 or eol.shape != (86,):
        raise ValueError("Expected the verified 86-cell C_full and T_eol inputs.")
    if not np.array_equal(cycles, 1 + 5 * np.arange(len(cycles))):
        raise ValueError("C_full columns must be five-cycle RPTs starting at cycle 1.")
    x0 = early_features(capacities)
    candidate_specs = [("neural_network", None)]
    candidate_specs.extend(
        (equation, weight)
        for equation in EQUATIONS
        for weight in config["physics_weight_grid"]
    )
    model_ids = [_candidate_id(equation, weight) for equation, weight in candidate_specs]
    model_ids.insert(0, "gp")
    predictions, fit_log, fold_metrics = [], [], []
    started = time.monotonic()

    for outer_fold in sorted(np.unique(fold_ids)):
        test = np.flatnonzero(fold_ids == outer_fold)
        train = np.flatnonzero(fold_ids != outer_fold)
        fold_record = {
            "fold": int(outer_fold),
            "train_cell_ids": ids[train].tolist(),
            "test_cell_ids": ids[test].tolist(),
            "models": {},
        }
        fold_rows = {model_id: [] for model_id in model_ids}

        gp_predictor, gp_record = tune_model(
            "gp", capacities, cycles, eol, train, x0, config, outer_fold
        )
        fold_record["models"]["gp"] = gp_record

        fitted_models = {}
        for equation, weight in candidate_specs:
            model_id = _candidate_id(equation, weight)
            fitted, training_record = fit_constitutive_model(
                None if equation == "neural_network" else equation,
                0.0 if weight is None else float(weight),
                capacities,
                cycles,
                eol,
                train,
                x0,
                config,
                seed=int(config["seed"]) + outer_fold * 100 + len(fitted_models),
            )
            fitted_models[model_id] = fitted
            fold_record["models"][model_id] = {
                "equation": equation,
                "physics_weight": weight,
                **training_record,
            }

        for cell in test:
            gp_as_function = lambda base, query, predictor=gp_predictor: (
                predictor(query_rows(base, query, config["maximum_predicted_rul_cycles"])),
                None,
            )
            for model_id, predictor in [("gp", gp_as_function), *fitted_models.items()]:
                score, parameters = _evaluate_curve(
                    predictor, x0[cell], cell, capacities, cycles, eol, config
                )
                actual_rul = score["actual_RUL_cycles_evaluation_only"]
                score.update(
                    {
                        "cell_id": int(ids[cell]),
                        "fold": int(outer_fold),
                        "method": model_id,
                        "RUL_absolute_error_cycles_lower_bound_if_censored_evaluation_only": abs(
                            score["predicted_RUL_cycles"] - actual_rul
                        ),
                        "parameters_predicted_from_early_input": (
                            json.dumps(
                                {
                                    name: float(value)
                                    for name, value in zip(
                                        LAMBDA_FIELD_NAMES.get(model_id.split("_lambda_")[0][5:]),
                                        parameters,
                                    )
                                },
                                sort_keys=True,
                            )
                            if parameters is not None
                            else None
                        ),
                    }
                )
                predictions.append(score)
                fold_rows[model_id].append(score)

        for model_id, rows in fold_rows.items():
            fold_metrics.append(
                {
                    "fold": int(outer_fold),
                    "method": model_id,
                    "n_test_cells": len(rows),
                    "capacity_trajectory_MAE_Ah": float(
                        np.mean(
                            [
                                row["capacity_trajectory_MAE_Ah_evaluation_only"]
                                for row in rows
                            ]
                        )
                    ),
                    "RUL_MAE_cycles_capped_lower_bound": float(
                        np.mean(
                            [
                                row[
                                    "RUL_absolute_error_cycles_lower_bound_if_censored_evaluation_only"
                                ]
                                for row in rows
                            ]
                        )
                    ),
                    "horizon_censored_predictions": sum(
                        row["predicted_RUL_status"] == "horizon_censored" for row in rows
                    ),
                }
            )
        fit_log.append(fold_record)
        print(
            f"fold={outer_fold} complete; elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    prediction_frame = pd.DataFrame(predictions)
    assignments = {
        "current_only": current_only_assignment(capacities, config["pack_size"]).tolist()
    }
    for model_id in model_ids:
        subset = prediction_frame[prediction_frame.method == model_id]
        rul = subset.set_index("cell_id")["predicted_RUL_cycles"].reindex(ids).to_numpy()
        if not np.isfinite(rul).all():
            raise ValueError(f"{model_id} did not predict finite RUL for every cell.")
        assignments[model_id] = rank_predicted_rul(
            rul, capacities[:, 1], ids, config["pack_size"]
        ).tolist()

    current_value = total_value(
        capacities,
        config["diagnosis_cycles"],
        assignments["current_only"],
        config["eol_capacity_Ah"],
    )
    metric_rows = []
    for model_id, packs in assignments.items():
        value = total_value(
            capacities, config["diagnosis_cycles"], packs, config["eol_capacity_Ah"]
        )
        if model_id == "current_only":
            capacity_mae = rul_mae = np.nan
            censored = 0
        else:
            subset = prediction_frame[prediction_frame.method == model_id]
            capacity_mae = float(
                subset["capacity_trajectory_MAE_Ah_evaluation_only"].mean()
            )
            rul_mae = float(
                subset[
                    "RUL_absolute_error_cycles_lower_bound_if_censored_evaluation_only"
                ].mean()
            )
            censored = int(
                (subset["predicted_RUL_status"] == "horizon_censored").sum()
            )
        equation, weight = (
            ("current_only", None)
            if model_id == "current_only"
            else (
                ("gp", None)
                if model_id == "gp"
                else (
                    ("neural_network", None)
                    if model_id == "neural_network"
                    else next(
                        (eq, float(lam))
                        for eq in EQUATIONS
                        for lam in config["physics_weight_grid"]
                        if model_id == _candidate_id(eq, lam)
                    )
                )
            )
        )
        metric_rows.append(
            {
                "method": model_id,
                "equation": equation,
                "physics_residual_weight": weight,
                "diagnostic_cycles_per_cell": int(config["diagnosis_cycles"]),
                "total_diagnostic_cost_cell_cycles": int(
                    config["diagnosis_cycles"] * len(ids)
                ),
                "N_inventory": len(ids),
                "pack_size_series": int(config["pack_size"]),
                "n_packs": len(packs),
                "n_unused": len(ids) - np.asarray(packs).size,
                "capacity_trajectory_MAE_Ah": capacity_mae,
                "RUL_MAE_cycles_capped_lower_bound": rul_mae,
                "horizon_censored_predictions": censored,
                "actual_V_Ah_cycles": value,
                "actual_V_delta_vs_current_only_Ah_cycles": value - current_value,
                "relative_V_index_current_only_100": 100.0 * value / current_value,
                "pack_value_scope": (
                    "exploratory_86_cell_OOF_not_independent_inventory_generalization"
                ),
                "human_selection": "no_winner_selected_review_pareto_tradeoffs",
                "oracle_regret": None,
            }
        )

    out = _input_path(config["result_dir"])
    out.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    assignments_payload = {
        method: {
            "pack_cell_ids": [[int(ids[index]) for index in pack] for pack in packs],
            "unused_cell_ids": [
                int(index_id)
                for index_id in ids[
                    np.setdiff1d(np.arange(len(ids)), np.asarray(packs).ravel())
                ]
            ],
        }
        for method, packs in assignments.items()
    }
    (out / "pack_assignments.json").write_text(
        json.dumps(assignments_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "fit_log.json").write_text(
        json.dumps(fit_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fold_metadata_path = fold_path.with_name("folds.metadata.json")
    manifest = {
        "experiment": config["experiment"],
        "config": config,
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (capacity_path, eol_path, fold_path)
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold_metadata": (
            json.loads(fold_metadata_path.read_text(encoding="utf-8"))
            if fold_metadata_path.exists()
            else {"status": "metadata_not_available"}
        ),
        "environment": {
            "python": sys.version,
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scikit-learn", "torch")
            },
        },
        "equations": {
            "sqrt_linear": (
                "Q=Q6-ks*(sqrt(x+eps)-sqrt(eps))-kl*x; "
                "dQ/dx+ks/(2*sqrt(x+eps))+kl=0"
            ),
            "exponential_linear": (
                "Q=Q6-kexp*(1-exp(-x/tau))-kl*x; "
                "dQ/dx+(kexp/tau)*exp(-x/tau)+kl=0"
            ),
        },
        "information_boundary": (
            "Evaluation predictors receive cycle-1 and cycle-6 capacity only. "
            "Evaluation future capacities and T_eol are accessed only after forecasts "
            "are fixed, for trajectory/RUL scoring and actual-V evaluation."
        ),
        "selection_policy": (
            "All configured equation/physics-weight candidates are reported; no winner is "
            "selected automatically. Full-inventory OOF V is exploratory."
        ),
        "oracle_regret": "not_computed_global_oracle_procedure_unresolved",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print(f"OUTPUT: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/constitutive_pinn_n10_8S_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_experiment(config)


if __name__ == "__main__":
    main()
