"""V2: held-out cell evaluation, calibrated supervised RUL, and pack swaps.

Development 43 cells = proper training 32 + calibration 11. Test 43 cells are
never used for feature scaling, tuning, model-family selection, or calibration.
All methods use exactly the same observation prefixes and pack constraints.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .real86 import allocation, posterior, rul_samples, score

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ['current_cycle', 'first_capacity', 'current_capacity', 'capacity_drop',
                 'overall_slope', 'first_three_slope', 'recent_three_slope',
                 'slope_change', 'linear_residual_sd', 'last_capacity_change']


def features(cycles, prefixes):
    """Only supplied prefixes enter features: no EOL, protocol or full-curve metadata."""
    x = np.asarray(cycles, dtype=float)
    q = np.asarray(prefixes, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(x) or len(x) < 3 or not np.isfinite(q).all():
        raise ValueError('Require at least three complete observed RPTs per cell.')
    def slope(local_x, local_q):
        centered = local_x - local_x.mean()
        return local_q @ centered / (centered @ centered)
    whole = slope(x, q)
    early, recent = slope(x[:3], q[:, :3]), slope(x[-3:], q[:, -3:])
    fitted = q.mean(axis=1)[:, None] + whole[:, None] * (x - x.mean())
    return np.column_stack([np.full(len(q), x[-1]), q[:, 0], q[:, -1], q[:, 0] - q[:, -1],
                            whole, early, recent, recent - early,
                            np.sqrt(np.mean((q - fitted) ** 2, axis=1)), q[:, -1] - q[:, -2]])


def make_estimator(kind, parameter):
    if kind == 'ridge_log':
        return Ridge(alpha=float(parameter))
    return GaussianProcessRegressor(
        kernel=RBF(parameter['length_scale'], length_scale_bounds='fixed') +
               WhiteKernel(parameter['noise_level'], noise_level_bounds='fixed'),
        alpha=1e-8, normalize_y=True, optimizer=None)


def tune_and_fit(kind, x, log_y, config, seed):
    candidates = config['ridge_alphas'] if kind == 'ridge_log' else config['gp_candidates']
    folds = list(KFold(config['inner_folds'], shuffle=True, random_state=seed).split(x))
    trials = []
    for parameter in candidates:
        errors = []
        for train, valid in folds:
            scaler = StandardScaler().fit(x[train])
            estimator = make_estimator(kind, parameter)
            estimator.fit(scaler.transform(x[train]), log_y[train])
            predicted = estimator.predict(scaler.transform(x[valid]))
            errors.extend(np.square(predicted - log_y[valid]))
        trials.append({'parameter': parameter, 'inner_log_RMSE': float(np.sqrt(np.mean(errors)))})
    best = min(trials, key=lambda r: r['inner_log_RMSE'])
    scaler = StandardScaler().fit(x)
    estimator = make_estimator(kind, best['parameter'])
    estimator.fit(scaler.transform(x), log_y)
    residual = log_y - estimator.predict(scaler.transform(x))
    raw_sd = max(config['minimum_log_sd'], float(np.sqrt(np.mean(residual ** 2))))
    return {'kind': kind, 'scaler': scaler, 'estimator': estimator, 'raw_sd': raw_sd,
            'inner_log_RMSE': best['inner_log_RMSE'], 'parameter': best['parameter'], 'trials': trials}


def predict_log(fitted, x, min_sd):
    scaled = fitted['scaler'].transform(x)
    if fitted['kind'] == 'gp_log':
        mean, covariance = fitted['estimator'].predict(scaled, return_cov=True)
        covariance += np.diag(np.maximum(0, min_sd ** 2 - np.diag(covariance)))
    else:
        mean = fitted['estimator'].predict(scaled)
        covariance = np.eye(len(x)) * max(min_sd, fitted['raw_sd']) ** 2
    return mean, covariance


def calibration_scale(cal_log_truth, cal_mean, cal_sd, coverage):
    """Split-calibration radius, fitted only on the reserved development cells."""
    residual_scores = np.abs(cal_log_truth - cal_mean) / cal_sd
    k = int(np.ceil((len(residual_scores) + 1) * coverage))
    if k > len(residual_scores):
        raise ValueError('Too few calibration cells for this finite-sample coverage level.')
    quantile = float(np.sort(residual_scores)[k - 1])
    return quantile / norm.ppf((1 + coverage) / 2), quantile


def sample_rul(log_mean, log_cov, normals, cap):
    root = np.linalg.cholesky(log_cov + np.eye(len(log_mean)) * 1e-10)
    logs = log_mean + normals @ root.T
    return np.exp(np.clip(logs, -30, np.log(cap)))


def optimize_packs(scenarios, initial, maximum_swaps):
    """Best-improvement swaps, including spare cells. Optimize sample utility only."""
    scenarios = np.asarray(scenarios)
    packs = np.asarray(initial).copy()
    n = scenarios.shape[1]
    objective = [float(score(scenarios, packs).mean())]
    for _ in range(maximum_swaps):
        values = scenarios[:, packs]
        minimum = values.min(axis=2)
        first_two = np.partition(values, 1, axis=2)[:, :, :2]
        second = first_two.max(axis=2)
        argmin = values.argmin(axis=2)
        membership = np.full(n, -1)
        membership[packs.ravel()] = np.repeat(np.arange(len(packs)), packs.shape[1])
        slots = {int(cell): (p, k) for p, pack in enumerate(packs) for k, cell in enumerate(pack)}
        # delta[i,j] is the change in expected value of i's pack if j replaces i.
        delta = np.zeros((n, n))
        for i, (p, k) in slots.items():
            except_i = np.where(argmin[:, p] == k, second[:, p], minimum[:, p])
            delta[i] = (np.minimum(except_i[:, None], scenarios) - minimum[:, p, None]).mean(axis=0)
        best_gain, best_pair = 1e-9, None
        for i in range(n):
            for j in range(i + 1, n):
                if membership[i] == membership[j]:
                    continue
                gain = (delta[i, j] if membership[i] >= 0 else 0) + (delta[j, i] if membership[j] >= 0 else 0)
                if gain > best_gain:
                    best_gain, best_pair = gain, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        before_i, before_j = slots.get(i), slots.get(j)
        if before_i is not None:
            packs[before_i] = j
        if before_j is not None:
            packs[before_j] = i
        objective.append(float(score(scenarios, packs).mean()))
        if objective[-1] < objective[-2] - 1e-7:
            raise AssertionError('A swap reduced its own sample objective.')
    return packs, objective


def evaluate(point, interval, actual, packs, oracle, ids, context, scenarios=None):
    value = float(score(actual, packs))
    low, high = interval if interval is not None else (None, None)
    prediction_valid = point is not None
    correlation = float(spearmanr(point, actual).statistic) if prediction_valid and np.std(point) > 0 else np.nan
    return {**context, 'utility_pack_cycles': value, 'oracle_pack_cycles': oracle,
            'score_100': 100 * value / oracle,
            'RUL_MAE': float(np.mean(np.abs(point - actual))) if prediction_valid else np.nan,
            'rank_spearman': correlation,
            'coverage80': float(np.mean((actual >= low) & (actual <= high))) if low is not None else np.nan,
            'interval80_width': float(np.mean(high - low)) if low is not None else np.nan,
            'predicted_utility_validation': float(score(scenarios, packs).mean()) if scenarios is not None else np.nan,
            'assignment_cell_ids': json.dumps(ids[packs].tolist()),
            'spare_cell_ids': json.dumps(ids[np.setdiff1d(np.arange(len(ids)), packs.ravel())].tolist())}


def add_case(capacities, cycles, ids, eol, proper, calibration, test, nobs, seed, fold, config):
    last = float(cycles[nobs - 1])
    # Prefix slicing occurs before feature extraction. Evaluation truth remains separate.
    x_train = features(cycles[:nobs], capacities[proper, :nobs])
    x_cal = features(cycles[:nobs], capacities[calibration, :nobs])
    x_test = features(cycles[:nobs], capacities[test, :nobs])
    y_train = np.log(eol[proper] - last)
    y_cal = np.log(eol[calibration] - last)
    actual = eol[test] - last
    test_ids = ids[test]
    oracle = float(score(actual, allocation(actual, config['pack_size'])))
    ctx = {'split_seed': seed, 'fold': fold, 'n_observations': nobs,
           'current_cycle': int(last), 'n_test_cells': len(test),
           'additional_cell_cycles': int(len(test) * (last - cycles[config['observation_counts'][0] - 1])),
           'total_test_cell_cycles': int(len(test) * last), 'n_packs': len(test) // config['pack_size'],
           'n_spares': len(test) % config['pack_size']}
    rows, predictions, fitted_log = [], [], []
    cap_packs = allocation(capacities[test, nobs - 1], config['pack_size'])
    rows.append(evaluate(None, None, actual, cap_packs, oracle, test_ids,
                         {**ctx, 'model': 'capacity', 'distribution': 'none', 'assembly': 'sort'}))
    rng = np.random.default_rng(seed * 10000 + fold * 1000 + nobs)
    normals = rng.normal(size=(config['optimization_scenarios'], len(test)))
    validation_normals = rng.normal(size=(config['validation_scenarios'], len(test)))
    linear_normals = rng.normal(size=(len(test), 256, 2))
    for name, window in [('linear_recent8', 8), ('linear_all', nobs)]:
        model = {**config['linear_baseline'], 'window_observations': window}
        samples = []
        for j, i in enumerate(test):
            mu, cov = posterior(cycles[:nobs], capacities[i, :nobs], model)
            samples.append(rul_samples(mu, cov, last, linear_normals[j], model, 1.44))
        samples = np.array(samples).T
        point = samples.mean(axis=0)
        interval = np.quantile(samples, [.1, .9], axis=0)
        rows.append(evaluate(point, interval, actual, allocation(point, config['pack_size']), oracle,
                             test_ids, {**ctx, 'model': name, 'distribution': 'raw', 'assembly': 'sort'}))
    learned = {}
    for kind in ['ridge_log', 'gp_log']:
        fitted = tune_and_fit(kind, x_train, y_train, config, seed + fold * 100 + nobs)
        cal_mean, cal_cov = predict_log(fitted, x_cal, config['minimum_log_sd'])
        factor, radius = calibration_scale(y_cal, cal_mean, np.sqrt(np.diag(cal_cov)), config['target_coverage'])
        log_mean, log_cov = predict_log(fitted, x_test, config['minimum_log_sd'])
        point = np.exp(np.clip(log_mean, -30, np.log(config['maximum_RUL_cycles'])))
        fitted_log.append({**ctx, 'model': kind, 'parameter': fitted['parameter'], 'trials': fitted['trials'],
                           'inner_log_RMSE': fitted['inner_log_RMSE'], 'calibration_sd_factor': factor,
                           'calibration_radius': radius, 'proper_training_cell_ids': ids[proper].tolist(),
                           'calibration_cell_ids': ids[calibration].tolist(), 'test_cell_ids': test_ids.tolist()})
        learned[kind] = {'inner_log_RMSE': fitted['inner_log_RMSE'], 'rows': []}
        for dist, scale in [('raw', 1.), ('calibrated', factor)]:
            covariance = log_cov * scale ** 2
            opt_samples = sample_rul(log_mean, covariance, normals, config['maximum_RUL_cycles'])
            valid_samples = sample_rul(log_mean, covariance, validation_normals, config['maximum_RUL_cycles'])
            sd = np.sqrt(np.diag(covariance))
            z = norm.ppf(.9)
            interval = np.exp(np.clip(np.array([log_mean - z * sd, log_mean + z * sd]), -30,
                                      np.log(config['maximum_RUL_cycles'])))
            initial = allocation(point, config['pack_size'])
            row = evaluate(point, interval, actual, initial, oracle, test_ids,
                           {**ctx, 'model': kind, 'distribution': dist, 'assembly': 'sort'}, valid_samples)
            row.update({'swaps': 0, 'optimization_gain': 0.0, 'validation_expected_gain': 0.0})
            rows.append(row)
            if dist == 'calibrated':
                learned[kind]['rows'].append(row)
                changed, objectives = optimize_packs(opt_samples, initial, config['maximum_swaps'])
                row = evaluate(point, interval, actual, changed, oracle, test_ids,
                               {**ctx, 'model': kind, 'distribution': dist, 'assembly': 'swap'}, valid_samples)
                row.update({'swaps': len(objectives) - 1, 'optimization_gain': objectives[-1] - objectives[0],
                            'validation_expected_gain': float((score(valid_samples, changed) - score(valid_samples, initial)).mean())})
                rows.append(row)
                learned[kind]['rows'].append(row)
            for j, cid in enumerate(test_ids):
                predictions.append({**ctx, 'cell_id': int(cid), 'model': kind, 'distribution': dist,
                                    'predicted_RUL_median': float(point[j]), 'interval80_low': float(interval[0, j]),
                                    'interval80_high': float(interval[1, j]), 'actual_RUL_evaluation_only': float(actual[j])})
    # Family selection is made using inner development errors, never test performance.
    winner = min(learned, key=lambda kind: learned[kind]['inner_log_RMSE'])
    for row in learned[winner]['rows']:
        rows.append({**row, 'model': 'dev_selected', 'selected_family': winner})
    return rows, predictions, fitted_log


def create_plots(frame, out):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = [('capacity', 'none', 'sort', 'Capacity sort'),
                ('linear_recent8', 'raw', 'sort', 'V1 local linear'),
                ('linear_all', 'raw', 'sort', 'Linear, all observations'),
                ('ridge_log', 'calibrated', 'sort', 'Ridge, sorted'),
                ('gp_log', 'calibrated', 'sort', 'GP, sorted'),
                ('dev_selected', 'calibrated', 'sort', 'Dev-selected, sorted'),
                ('dev_selected', 'calibrated', 'swap', 'Dev-selected + swaps')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for color_index, (model, dist, assembly, label) in enumerate(variants):
        subset = frame[(frame.model == model) & (frame.distribution == dist) & (frame.assembly == assembly)]
        for ax, metric in zip(axes, ['score_100', 'RUL_MAE']):
            if model == 'capacity' and metric == 'RUL_MAE':
                continue
            grouped = subset.groupby('current_cycle')[metric].mean()
            ax.plot(grouped.index, grouped.values, '-o', label=label, color=f'C{color_index}')
    axes[0].set(ylabel='Held-out pack score / 100', title='Matched test pools and observation budgets')
    axes[0].axhline(95, color='black', ls='--', alpha=.4)
    axes[1].set(ylabel='Held-out RUL MAE (cycles)', title='Point-prediction error')
    for ax in axes:
        ax.set_xlabel('Observed through cycle (all test cells equally)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('V2 | 32 train + 11 calibration + 43 test | 3 split seeds x 2 folds')
    fig.savefig(out / 'v2_model_comparison.png', dpi=180)
    plt.close(fig)
    subset = frame[(frame.assembly == 'sort') & frame.model.isin(['ridge_log', 'gp_log'])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
    for (model, dist), part in subset.groupby(['model', 'distribution']):
        grouped = part.groupby('current_cycle').coverage80.mean()
        axes[0].plot(grouped.index, grouped.values, '-o', label=f'{model}: {dist}')
        widths = part.groupby('current_cycle').interval80_width.mean()
        axes[1].plot(widths.index, widths.values, '-o', label=f'{model}: {dist}')
    axes[0].axhline(.8, ls='--', color='black', label='Nominal 80%')
    axes[0].set(ylabel='Empirical held-out coverage', ylim=(0, 1.03), title='Coverage')
    axes[1].set(ylabel='Mean 80% interval width (cycles)', title='Width tradeoff')
    for ax in axes:
        ax.set_xlabel('Observed through cycle')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Reserved calibration cells; test labels never set interval width')
    fig.savefig(out / 'v2_calibration.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/model_comparison_v2.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = Path(config['data_dir']).expanduser()
    data = (data if data.is_absolute() else ROOT / data).resolve()
    with np.load(data / 'C_full.npz', allow_pickle=False) as z:
        capacities, cycles, ids = z['C_full'].copy(), z['cycles'].copy(), z['cell_ids'].copy()
    eol = np.load(data / 'T_eol.npy')
    if np.load(data / 'censor_mask.npy').any():
        raise ValueError('Censored labels need a different training objective.')
    maxobs = max(config['observation_counts'])
    if len(ids) != 86 or not np.isfinite(capacities[:, :maxobs]).all() or cycles[maxobs - 1] >= eol.min():
        raise ValueError('Need the verified 86 complete early prefixes and positive RUL labels.')
    out = ROOT / 'results' / config['run_name']
    out.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), ROOT / 'battery_sim/real86.py', args.config.resolve()]
    source_labels = {p: p.resolve().relative_to(ROOT.parent).as_posix() for p in source_files}
    source_labels = {p: p.resolve().relative_to(ROOT.parent).as_posix() for p in source_files}
    source_labels = {p: p.resolve().relative_to(ROOT.parent).as_posix() for p in source_files}
    manifest = {'config': config, 'feature_names': FEATURE_NAMES,
                'environment': {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
                                for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}},
                'source_sha256': {source_labels[p]: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()
                                if p.name in ['C_full.npz', 'T_eol.npy', 'censor_mask.npy', 'meta.csv', 'cap_raw.parquet']},
                'point_prediction': 'Learned models use exp(predicted log RUL), the lognormal median; V1 uses its original sampled mean.',
                'calibration': 'Separate 11-cell split calibration in log-RUL space; 80% interval radius rescales Gaussian log uncertainty.',
                'scenario_dependence': 'GP uses joint predictive covariance; Ridge uses independent residual scenarios.',
                'limitations': ['Same 86-cell exploratory dataset; split repetitions are dependent.',
                                'This benchmark isolates prediction and assembly. All test cells get equal inspection effort.',
                                'No new adaptive inspection policy or stopping rule is evaluated in this version.',
                                'Scores must be compared to matched V1 and capacity baselines here, not the previous 86-cell pool.',
                                'The full dataset was inspected during earlier pilots; this is not a pristine external validation.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    all_rows, all_predictions, all_fits, splits = [], [], [], []
    started = time.monotonic()
    for seed in config['split_seeds']:
        outer = KFold(config['outer_folds'], shuffle=True, random_state=seed)
        for fold, (development, test) in enumerate(outer.split(ids)):
            order = np.random.default_rng(seed + fold * 100).permutation(development)
            calibration = np.sort(order[:config['calibration_cells']])
            proper = np.sort(order[config['calibration_cells']:])
            assert not set(proper) & set(calibration) and not set(development) & set(test)
            splits.append({'split_seed': seed, 'fold': fold, 'proper_train_ids': ids[proper].tolist(),
                           'calibration_ids': ids[calibration].tolist(), 'test_ids': ids[test].tolist()})
            for nobs in config['observation_counts']:
                rows, predictions, fits = add_case(capacities, cycles, ids, eol, proper, calibration, test,
                                                    nobs, seed, fold, config)
                all_rows.extend(rows); all_predictions.extend(predictions); all_fits.extend(fits)
                pd.DataFrame(all_rows).to_csv(out / 'metrics.csv', index=False)
                print(f'seed={seed} fold={fold} observed={cycles[nobs-1]}: completed; elapsed={time.monotonic()-started:.1f}s', flush=True)
    frame = pd.DataFrame(all_rows)
    pd.DataFrame(all_predictions).to_csv(out / 'predictions.csv', index=False)
    (out / 'splits.json').write_text(json.dumps(splits, indent=2) + '\n')
    (out / 'fitted_model_log.json').write_text(json.dumps(all_fits, ensure_ascii=False, indent=2) + '\n')
    summary = frame.groupby(['current_cycle', 'model', 'distribution', 'assembly']).agg(
        score_mean=('score_100', 'mean'), score_sd=('score_100', 'std'), MAE_mean=('RUL_MAE', 'mean'),
        coverage80_mean=('coverage80', 'mean'), interval_width_mean=('interval80_width', 'mean'),
        utility_mean=('utility_pack_cycles', 'mean'), validation_gain_mean=('validation_expected_gain', 'mean')).reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    create_plots(frame, out)
    print(summary[(summary.current_cycle.isin([21, 101]))].round(3).to_string(index=False), flush=True)
    print(f'OUTPUT: {out.relative_to(ROOT).as_posix()}', flush=True)


if __name__ == '__main__':
    main()
