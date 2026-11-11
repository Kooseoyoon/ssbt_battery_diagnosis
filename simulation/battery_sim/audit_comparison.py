"""Reconstruct V2 scores and audit train/calibration/test provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .real86 import allocation, score

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'configs/model_comparison_v2.json').read_text())
    out = ROOT / 'results' / config['run_name']
    data = Path(config['data_dir']).expanduser()
    data = (data if data.is_absolute() else ROOT / data).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, digest in manifest['data_sha256'].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    with np.load(data / 'C_full.npz') as z:
        ids = z['cell_ids']
    eol = np.load(data / 'T_eol.npy')
    eol_of = dict(zip(ids.tolist(), eol.tolist()))
    splits = json.loads((out / 'splits.json').read_text())
    split_of = {(r['split_seed'], r['fold']): r for r in splits}
    for row in splits:
        proper, cal, test = map(set, [row['proper_train_ids'], row['calibration_ids'], row['test_ids']])
        assert (len(proper), len(cal), len(test)) == (32, 11, 43)
        assert not proper & cal and not proper & test and not cal & test
        assert proper | cal | test == set(ids)
    for seed in config['split_seeds']:
        test_ids = [cid for r in splits if r['split_seed'] == seed for cid in r['test_ids']]
        assert sorted(test_ids) == sorted(ids.tolist())
    fits = json.loads((out / 'fitted_model_log.json').read_text())
    fit_of = {(r['split_seed'], r['fold'], r['n_observations'], r['model']): r for r in fits}
    for fit in fits:
        split = split_of[(fit['split_seed'], fit['fold'])]
        assert fit['proper_training_cell_ids'] == split['proper_train_ids']
        assert fit['calibration_cell_ids'] == split['calibration_ids']
        best = min(fit['trials'], key=lambda r: r['inner_log_RMSE'])
        assert fit['parameter'] == best['parameter']
        assert fit['inner_log_RMSE'] == best['inner_log_RMSE']
    metrics = pd.read_csv(out / 'metrics.csv')
    for row in metrics.itertuples():
        split = split_of[(row.split_seed, row.fold)]
        test = split['test_ids']
        position = {cid: i for i, cid in enumerate(test)}
        actual = np.array([max(0, eol_of[cid] - row.current_cycle) for cid in test])
        assignment_ids = json.loads(row.assignment_cell_ids)
        packs = np.array([[position[cid] for cid in pack] for pack in assignment_ids])
        spares = set(json.loads(row.spare_cell_ids))
        assert packs.shape == (5, 8) and len(set(packs.ravel())) == 40 and len(spares) == 3
        assert spares == set(test) - {cid for pack in assignment_ids for cid in pack}
        oracle = float(score(actual, allocation(actual, 8)))
        value = float(score(actual, packs))
        assert np.isclose(oracle, row.oracle_pack_cycles)
        assert np.isclose(value, row.utility_pack_cycles)
        assert np.isclose(100 * value / oracle, row.score_100)
        assert row.total_test_cell_cycles == 43 * row.current_cycle
        assert row.additional_cell_cycles == 43 * (row.current_cycle - 21)
        if row.model == 'dev_selected':
            family = min(['ridge_log', 'gp_log'], key=lambda m: fit_of[
                (row.split_seed, row.fold, row.n_observations, m)]['inner_log_RMSE'])
            assert row.selected_family == family
    predictions = pd.read_csv(out / 'predictions.csv')
    for (_, _, _, model), group in predictions.groupby(['split_seed', 'fold', 'n_observations', 'model']):
        raw = group[group.distribution == 'raw'].sort_values('cell_id')
        calibrated = group[group.distribution == 'calibrated'].sort_values('cell_id')
        assert np.array_equal(raw.predicted_RUL_median.to_numpy(), calibrated.predicted_RUL_median.to_numpy())
    for row in predictions.itertuples():
        assert np.isclose(row.actual_RUL_evaluation_only, eol_of[row.cell_id] - row.current_cycle)
        assert row.interval80_low <= row.interval80_high
    for keys, group in metrics[metrics.model.isin(['ridge_log', 'gp_log'])].groupby(
            ['split_seed', 'fold', 'n_observations', 'model', 'distribution']):
        pred = predictions[(predictions.split_seed == keys[0]) & (predictions.fold == keys[1]) &
                           (predictions.n_observations == keys[2]) & (predictions.model == keys[3]) &
                           (predictions.distribution == keys[4])]
        coverage = ((pred.actual_RUL_evaluation_only >= pred.interval80_low) &
                    (pred.actual_RUL_evaluation_only <= pred.interval80_high)).mean()
        mae = np.abs(pred.predicted_RUL_median - pred.actual_RUL_evaluation_only).mean()
        for row in group.itertuples():
            assert np.isclose(mae, row.RUL_MAE) and np.isclose(coverage, row.coverage80)
        swaps = group[group.assembly == 'swap']
        assert (swaps.optimization_gain >= -1e-7).all()
    report = {'status': 'passed', 'metric_rows_verified': len(metrics), 'prediction_rows_verified': len(predictions),
              'outer_evaluation_pools': len(splits), 'fitted_model_cases': len(fits),
              'checks': ['data and model-source SHA-256 unchanged', 'disjoint 32/11/43 cell partitions',
                         'every cell evaluated once per split seed', 'inner-development-only parameter selection',
                         'development-only model-family selection', '40 selected cells plus 3 spares',
                         'true deployment utility, oracle and score reconstructed', 'inspection cost reconstructed',
                         'calibration leaves point predictions fixed', 'MAE and interval coverage reconstructed']}
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    summary = pd.read_csv(out / 'summary.csv')
    deltas = []
    for cycle, group in summary.groupby('current_cycle'):
        def find(model, dist='calibrated', assembly='sort'):
            return group[(group.model == model) & (group.distribution == dist) & (group.assembly == assembly)].iloc[0]
        old, new, swapped = find('linear_recent8', 'raw'), find('dev_selected'), find('dev_selected', assembly='swap')
        capacity = find('capacity', 'none')
        deltas.append({'current_cycle': int(cycle), 'matched_V1_score': old.score_mean,
                       'capacity_baseline_score': capacity.score_mean, 'new_selected_score': new.score_mean,
                       'new_minus_V1_points': new.score_mean - old.score_mean,
                       'new_minus_capacity_points': new.score_mean - capacity.score_mean,
                       'V1_MAE': old.MAE_mean, 'new_MAE': new.MAE_mean,
                       'MAE_reduction_percent': 100 * (1 - new.MAE_mean / old.MAE_mean),
                       'swap_score': swapped.score_mean, 'swap_minus_sort_points': swapped.score_mean - new.score_mean})
    pd.DataFrame(deltas).to_csv(out / 'comparison_deltas.csv', index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(pd.DataFrame(deltas).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
