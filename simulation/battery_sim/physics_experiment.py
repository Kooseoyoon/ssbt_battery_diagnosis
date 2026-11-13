"""C_full-only n=10, 8S OOF comparison of GP, NN, and physics-regularized NN."""

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from .shared_folds import validate_folds
from .value_metrics import current_only_assignment, total_value

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("gp", "neural_network", "physics_neural_network")
ANCHOR_CYCLE = 6


def early_features(capacities):
    """Only cycle-1 and cycle-6 capacity, and their observed difference, are used."""
    c = np.asarray(capacities, dtype=float)
    if c.ndim != 2 or c.shape[1] < 2 or not np.isfinite(c[:, :2]).all():
        raise ValueError("Expected finite cycle-1 and cycle-6 capacities.")
    return np.column_stack((c[:, 0], c[:, 1], (c[:, 1] - c[:, 0]) / 5.0))


def _eol_column(cycles, eol_cycle):
    matches = np.flatnonzero(np.asarray(cycles) == eol_cycle)
    if len(matches) != 1:
        raise ValueError("Each EOL must match exactly one observed RPT cycle.")
    return int(matches[0])


def _validate_cell_history(capacity, cycles, eol_cycle, threshold):
    eol_column = _eol_column(cycles, eol_cycle)
    if eol_column < 2 or not np.isfinite(capacity[:eol_column + 1]).all():
        raise ValueError("A cell needs a finite post-diagnosis history through EOL.")
    below = np.flatnonzero(np.isfinite(capacity) & (capacity < threshold))
    if not len(below) or int(below[0]) != eol_column:
        raise ValueError("T_eol must identify the first observed capacity below threshold.")
    return eol_column


def trajectory_samples(capacities, cycles, eol, cell_indices, config):
    """Build training rows from each selected cell's disclosed long history."""
    c = np.asarray(capacities, dtype=float)
    x0 = early_features(c)
    anchor = int(np.flatnonzero(cycles == ANCHOR_CYCLE)[0])
    rows, values, owners = [], [], []
    for cell in np.asarray(cell_indices, dtype=int):
        end = _validate_cell_history(c[cell], cycles, eol[cell], config["eol_capacity_Ah"])
        future = np.arange(anchor + 1, end + 1, dtype=int)
        limit = int(config["trajectory_landmarks_per_cell"])
        if len(future) > limit:
            future = future[np.linspace(0, len(future) - 1, limit).round().astype(int)]
        selected = np.concatenate(([anchor], future))
        elapsed = (cycles[selected] - ANCHOR_CYCLE) / config["maximum_predicted_rul_cycles"]
        rows.extend(np.column_stack((np.repeat(x0[cell][None, :], len(selected), axis=0), elapsed)))
        values.extend(c[cell, selected])
        owners.extend([cell] * len(selected))
    return np.asarray(rows), np.asarray(values), np.asarray(owners, dtype=int)


def recovery_allowance(capacities, cycles, eol, cell_indices, config):
    increments = []
    anchor = int(np.flatnonzero(cycles == ANCHOR_CYCLE)[0])
    for cell in np.asarray(cell_indices, dtype=int):
        end = _validate_cell_history(capacities[cell], cycles, eol[cell], config["eol_capacity_Ah"])
        differences = np.diff(capacities[cell, anchor:end + 1])
        increments.extend(np.maximum(differences, 0.0))
    if not increments:
        raise ValueError("Cannot estimate a recovery allowance without training histories.")
    return float(np.quantile(increments, config["recovery_increment_quantile"]))


def query_rows(base_features, cycles, maximum_rul):
    base = np.asarray(base_features, dtype=float)
    t = (np.asarray(cycles, dtype=float) - ANCHOR_CYCLE) / maximum_rul
    return np.column_stack((np.repeat(base[None, :], len(t), axis=0), t))


def predicted_rul(curve, cycles, start_cycle, threshold, maximum_rul):
    """First predicted strict threshold crossing; an absent crossing is right-censored."""
    q, x = np.asarray(curve, dtype=float), np.asarray(cycles, dtype=float)
    if q.ndim != 1 or q.shape != x.shape or not np.isfinite(q).all():
        raise ValueError("Predicted capacity curve must be finite and aligned to cycles.")
    below = np.flatnonzero(q < threshold)
    if not len(below):
        return float(maximum_rul), "horizon_censored"
    rul = max(0.0, float(x[int(below[0])] - start_cycle))
    return rul, ("already_below_threshold" if rul == 0 else "crossing_found")


def rank_predicted_rul(rul, observed_capacity, cell_ids, pack_size):
    """Rank by predicted RUL; resolve equal/censored RULs by observed cycle-6 capacity."""
    rul = np.asarray(rul, dtype=float)
    capacity = np.asarray(observed_capacity, dtype=float)
    ids = np.asarray(cell_ids)
    if (
        rul.ndim != 1
        or capacity.shape != rul.shape
        or ids.shape != rul.shape
        or not np.isfinite(rul).all()
        or not np.isfinite(capacity).all()
        or len(np.unique(ids)) != len(ids)
        or not 1 <= int(pack_size) <= len(rul)
    ):
        raise ValueError("Expected finite RUL and observed capacity for unique inventory cells.")
    used = len(rul) // int(pack_size) * int(pack_size)
    order = np.lexsort((ids, -capacity, -rul))
    return order[:used].reshape(-1, int(pack_size))


def _make_network(input_width, hidden_units):
    return nn.Sequential(
        nn.Linear(input_width, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, 1),
    )


def _fit_predictor(kind, x_train, y_train, train_base, config, parameter, seed, allowance):
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
    x_scaled = x_scaler.transform(x_train)
    y_scaled = y_scaler.transform(np.asarray(y_train).reshape(-1, 1)).ravel()

    if kind == "gp":
        model = GaussianProcessRegressor(
            kernel=RBF(float(parameter["length_scale"]), length_scale_bounds="fixed")
            + WhiteKernel(float(config["gp_noise_level"]), noise_level_bounds="fixed"),
            alpha=1e-8,
            optimizer=None,
            normalize_y=False,
        )
        model.fit(x_scaled, y_scaled)

        def predict(rows):
            return y_scaler.inverse_transform(
                model.predict(x_scaler.transform(rows)).reshape(-1, 1)
            ).ravel()

        return predict

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    model = _make_network(x_train.shape[1], int(config["nn_hidden_units"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["nn_learning_rate"]),
        weight_decay=float(parameter["weight_decay"]),
    )
    tx = torch.as_tensor(x_scaled, dtype=torch.float32)
    ty = torch.as_tensor(y_scaled[:, None], dtype=torch.float32)
    mean = torch.as_tensor(y_scaler.mean_[0], dtype=torch.float32)
    scale = torch.as_tensor(y_scaler.scale_[0], dtype=torch.float32)

    if kind == "physics_neural_network":
        grid_cycles = ANCHOR_CYCLE + np.linspace(
            0.0, config["maximum_predicted_rul_cycles"] + 5.0, 41
        )
        grid_time = (grid_cycles - ANCHOR_CYCLE) / config["maximum_predicted_rul_cycles"]
        raw_grid = np.concatenate(
            (
                np.repeat(np.asarray(train_base)[:, None, :], len(grid_time), axis=1),
                np.broadcast_to(grid_time, (len(train_base), len(grid_time)))[:, :, None],
            ),
            axis=2,
        ).reshape(-1, x_train.shape[1])
        grid_x = torch.as_tensor(x_scaler.transform(raw_grid), dtype=torch.float32)
        recovery_limit = torch.as_tensor(
            float(allowance) * np.diff(grid_cycles) / 5.0, dtype=torch.float32
        )
        capacity_scale = float(config["physics_capacity_scale_Ah"])
        physics_weight = float(parameter["physics_weight"])

    for _ in range(int(config["nn_epochs"])):
        optimizer.zero_grad()
        fit_loss = torch.mean((model(tx).reshape(-1) - ty.reshape(-1)) ** 2)
        loss = fit_loss
        if kind == "physics_neural_network":
            forecast = (model(grid_x).reshape(len(train_base), -1) * scale) + mean
            increments = forecast[:, 1:] - forecast[:, :-1]
            grid_step_rpts = float(np.diff(grid_cycles)[0] / 5.0)
            curvature = (
                forecast[:, 2:] - 2.0 * forecast[:, 1:-1] + forecast[:, :-2]
            ) / grid_step_rpts**2
            monotonic_loss = torch.relu(increments - recovery_limit).square().mean() / capacity_scale**2
            smooth_loss = curvature.square().mean() / capacity_scale**2
            loss = fit_loss + physics_weight * (
                monotonic_loss + float(config["smoothness_weight"]) * smooth_loss
            )
        loss.backward()
        optimizer.step()

    def predict(rows):
        with torch.no_grad():
            scaled_rows = torch.as_tensor(x_scaler.transform(rows), dtype=torch.float32)
            output = model(scaled_rows).cpu().numpy().reshape(-1, 1)
        return y_scaler.inverse_transform(output).ravel()

    return predict


def _validation_metrics(predict, capacities, cycles, eol, x0, cells, config):
    capacity_errors, log_rul_errors = [], []
    start_cycle = int(config["diagnosis_cycles"]) + 1
    horizon = start_cycle + 5 * np.arange(
        int(config["maximum_predicted_rul_cycles"] // 5) + 1
    )
    for cell in cells:
        end = _validate_cell_history(capacities[cell], cycles, eol[cell], config["eol_capacity_Ah"])
        selected = np.arange(int(np.flatnonzero(cycles == ANCHOR_CYCLE)[0]) + 1, end + 1)
        rows = query_rows(x0[cell], cycles[selected], config["maximum_predicted_rul_cycles"])
        capacity_errors.append(float(np.mean(np.abs(predict(rows) - capacities[cell, selected]))))
        forecast = predict(query_rows(x0[cell], horizon, config["maximum_predicted_rul_cycles"]))
        predicted, _ = predicted_rul(
            forecast, horizon, start_cycle, config["eol_capacity_Ah"],
            config["maximum_predicted_rul_cycles"],
        )
        actual = float(eol[cell] - start_cycle)
        log_rul_errors.append(float(np.log(max(predicted, 1.0)) - np.log(actual)))
    return {
        "capacity_MAE_Ah": float(np.mean(capacity_errors)),
        "log_RUL_RMSE": float(np.sqrt(np.mean(np.square(log_rul_errors)))),
    }


def tune_model(kind, capacities, cycles, eol, outer_train, x0, config, outer_fold):
    inner = KFold(
        n_splits=int(config["inner_folds"]),
        shuffle=True,
        random_state=int(config["inner_seed"]) + int(outer_fold),
    )
    candidates = []
    if kind == "gp":
        candidates = [{"length_scale": value} for value in config["gp_length_scales"]]
    else:
        for decay in config["nn_weight_decays"]:
            if kind == "neural_network":
                candidates.append({"weight_decay": decay})
            else:
                for weight in config["physics_weights"]:
                    candidates.append({"weight_decay": decay, "physics_weight": weight})
    trials = []
    for candidate_index, parameter in enumerate(candidates):
        fold_errors = []
        for inner_fold, (train_pos, valid_pos) in enumerate(inner.split(outer_train)):
            train_cells, valid_cells = outer_train[train_pos], outer_train[valid_pos]
            train_x, train_y, _ = trajectory_samples(capacities, cycles, eol, train_cells, config)
            allowance = recovery_allowance(capacities, cycles, eol, train_cells, config)
            fitted = _fit_predictor(
                kind, train_x, train_y, x0[train_cells], config, parameter,
                seed=10000 + outer_fold * 1000 + inner_fold * 100 + candidate_index,
                allowance=allowance,
            )
            fold_errors.append(
                _validation_metrics(fitted, capacities, cycles, eol, x0, valid_cells, config)
            )
        trials.append(
            {
                "parameters": parameter,
                "inner_log_RUL_RMSE": float(np.mean([row["log_RUL_RMSE"] for row in fold_errors])),
                "inner_capacity_MAE_Ah": float(np.mean([row["capacity_MAE_Ah"] for row in fold_errors])),
                "inner_folds": fold_errors,
            }
        )
    best = min(trials, key=lambda row: (row["inner_log_RUL_RMSE"], row["inner_capacity_MAE_Ah"]))
    train_x, train_y, _ = trajectory_samples(capacities, cycles, eol, outer_train, config)
    allowance = recovery_allowance(capacities, cycles, eol, outer_train, config)
    fitted = _fit_predictor(
        kind, train_x, train_y, x0[outer_train], config, best["parameters"],
        seed=20000 + int(outer_fold), allowance=allowance,
    )
    return fitted, {
        "parameters": best["parameters"],
        "inner_log_RUL_RMSE": best["inner_log_RUL_RMSE"],
        "inner_capacity_MAE_Ah": best["inner_capacity_MAE_Ah"],
        "trials": trials,
        "training_recovery_allowance_Ah_per_RPT": allowance,
    }


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
    if capacities.shape[0] != 86 or capacities.shape[1] != len(cycles) or eol.shape != (86,):
        raise ValueError("Expected the verified 86-cell C_full and T_eol inputs.")
    if not np.array_equal(cycles, 1 + 5 * np.arange(len(cycles))):
        raise ValueError("C_full columns must be five-cycle RPTs starting at cycle 1.")
    anchor = int(np.flatnonzero(cycles == ANCHOR_CYCLE)[0])
    start_cycle = int(config["diagnosis_cycles"]) + 1
    if cycles[anchor + 1] != start_cycle:
        raise ValueError("The cycle-6 observation must precede the cycle-11 evaluation start.")
    x0 = early_features(capacities)
    predictions, fit_log, fold_metrics = [], [], []
    started = time.monotonic()

    for outer_fold in sorted(np.unique(fold_ids)):
        test = np.flatnonzero(fold_ids == outer_fold)
        train = np.flatnonzero(fold_ids != outer_fold)
        fold_fit = {"fold": int(outer_fold), "train_cell_ids": ids[train].tolist(),
                    "test_cell_ids": ids[test].tolist(), "models": {}}
        fold_prediction = {}
        for kind in MODEL_NAMES:
            predictor, selected = tune_model(kind, capacities, cycles, eol, train, x0, config, outer_fold)
            fold_fit["models"][kind] = selected
            for cell in test:
                horizon_cycles = start_cycle + 5 * np.arange(
                    int(config["maximum_predicted_rul_cycles"] // 5) + 1
                )
                forecast = predictor(
                    query_rows(x0[cell], horizon_cycles, config["maximum_predicted_rul_cycles"])
                )
                rul, status = predicted_rul(
                    forecast, horizon_cycles, start_cycle, config["eol_capacity_Ah"],
                    config["maximum_predicted_rul_cycles"],
                )
                end = _validate_cell_history(
                    capacities[cell], cycles, eol[cell], config["eol_capacity_Ah"]
                )
                observed_future = np.arange(anchor + 1, end + 1)
                actual_cycles = cycles[observed_future]
                actual_forecast = predictor(
                    query_rows(x0[cell], actual_cycles, config["maximum_predicted_rul_cycles"])
                )
                capacity_mae = float(np.mean(np.abs(actual_forecast - capacities[cell, observed_future])))
                actual_rul = float(eol[cell] - start_cycle)
                row = {
                    "cell_id": int(ids[cell]),
                    "fold": int(outer_fold),
                    "method": kind,
                    "predicted_RUL_cycles": rul,
                    "predicted_RUL_status": status,
                    "actual_RUL_cycles_evaluation_only": actual_rul,
                    "RUL_absolute_error_cycles_lower_bound_if_censored_evaluation_only": abs(
                        rul - actual_rul
                    ),
                    "capacity_trajectory_MAE_Ah_evaluation_only": capacity_mae,
                    "actual_T_eol_evaluation_only": int(eol[cell]),
                    "predicted_capacity_at_cycle_11_Ah": float(forecast[0]),
                }
                predictions.append(row)
                fold_prediction.setdefault(kind, []).append(row)
        fit_log.append(fold_fit)
        for kind, rows in fold_prediction.items():
            fold_metrics.append(
                {
                    "fold": int(outer_fold),
                    "method": kind,
                    "n_test_cells": len(rows),
                    "capacity_trajectory_MAE_Ah": float(np.mean(
                        [row["capacity_trajectory_MAE_Ah_evaluation_only"] for row in rows]
                    )),
                    "RUL_MAE_cycles_capped_lower_bound": float(np.mean(
                        [row["RUL_absolute_error_cycles_lower_bound_if_censored_evaluation_only"]
                         for row in rows]
                    )),
                    "horizon_censored_predictions": sum(
                        row["predicted_RUL_status"] == "horizon_censored" for row in rows
                    ),
                }
            )
        print(
            f"fold={outer_fold} complete; elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    prediction_frame = pd.DataFrame(predictions)
    predicted_rul_by_method = {
        kind: prediction_frame.loc[prediction_frame.method == kind]
        .set_index("cell_id")["predicted_RUL_cycles"]
        .reindex(ids)
        .to_numpy()
        for kind in MODEL_NAMES
    }
    assignments = {"current_only": current_only_assignment(capacities, config["pack_size"]).tolist()}
    for kind, rul in predicted_rul_by_method.items():
        if not np.isfinite(rul).all():
            raise ValueError(f"{kind} did not produce a finite OOF RUL for every inventory cell.")
        assignments[kind] = rank_predicted_rul(
            rul, capacities[:, 1], ids, config["pack_size"]
        ).tolist()

    current_value = total_value(
        capacities, config["diagnosis_cycles"], assignments["current_only"],
        config["eol_capacity_Ah"],
    )
    metric_rows = []
    for method, packs in assignments.items():
        value = total_value(capacities, config["diagnosis_cycles"], packs, config["eol_capacity_Ah"])
        if method == "current_only":
            trajectory_mae = rul_mae = np.nan
            censored = 0
        else:
            subset = prediction_frame[prediction_frame.method == method]
            trajectory_mae = float(subset["capacity_trajectory_MAE_Ah_evaluation_only"].mean())
            rul_mae = float(
                subset[
                    "RUL_absolute_error_cycles_lower_bound_if_censored_evaluation_only"
                ].mean()
            )
            censored = int((subset["predicted_RUL_status"] == "horizon_censored").sum())
        metric_rows.append(
            {
                "method": method,
                "diagnostic_cycles_per_cell": int(config["diagnosis_cycles"]),
                "total_diagnostic_cost_cell_cycles": int(config["diagnosis_cycles"] * len(ids)),
                "N_inventory": len(ids),
                "pack_size_series": int(config["pack_size"]),
                "n_packs": len(packs),
                "n_unused": len(ids) - np.asarray(packs).size,
                "capacity_trajectory_MAE_Ah": trajectory_mae,
                "RUL_MAE_cycles_capped_lower_bound": rul_mae,
                "horizon_censored_predictions": censored,
                "actual_V_Ah_cycles": value,
                "actual_V_delta_vs_current_only_Ah_cycles": value - current_value,
                "pack_value_scope": "exploratory_86_cell_OOF_not_independent_inventory_generalization",
                "oracle_regret": None,
            }
        )

    out = _input_path(config["result_dir"])
    out.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    assignment_payload = {
        method: {
            "pack_cell_ids": [[int(ids[index]) for index in pack] for pack in packs],
            "unused_cell_ids": [
                int(ids[index]) for index in np.setdiff1d(np.arange(len(ids)), np.asarray(packs).ravel())
            ],
        }
        for method, packs in assignments.items()
    }
    (out / "pack_assignments.json").write_text(
        json.dumps(assignment_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
        "information_boundary": (
            "Evaluation cells expose only cycle-1 and cycle-6 capacity to the predictors. "
            "Their future C_full and T_eol are accessed only after OOF forecasts are fixed "
            "for trajectory/RUL scoring and actual-V evaluation."
        ),
        "preprocessing_and_selection": (
            "Input/target scalers and GP/NN hyperparameter selection are refit within each "
            "outer training fold; inner validation splits physical cells."
        ),
        "pack_value_scope": (
            "Full-inventory OOF pack value is exploratory; each cell is predicted out of fold, "
            "but different folds' models have trained on other inventory cells."
        ),
        "oracle_regret": "not_computed_global_oracle_procedure_unresolved",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print(f"OUTPUT: {out}", flush=True)
    return out


def _input_path(path):
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/physics_capacity_experiment_v1.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_experiment(config)


if __name__ == "__main__":
    main()
