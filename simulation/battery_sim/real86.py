"""Pilot on 86 real capacity trajectories, with observation-limited decisions.

No EOL labels are used to fit or select inspections. A local Bayesian linear
capacity model is deliberately used as an untrained starting baseline. Physical
interaction effects after assembling a pack are outside this replay model.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def allocation(values, pack_size):
    """Use the best floor(N/m)*m predicted cells; all other cells are spares.

    For known scalar lifetimes this maximizes sum of equal-pack minima with
    unrestricted subset choice. Under uncertainty it is a mean-sorting baseline.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Expected finite, nonnegative lifetimes.")
    if pack_size < 1:
        raise ValueError("Pack size must be positive.")
    count = values.shape[-1] // pack_size * pack_size
    if count == 0:
        raise ValueError("Not enough cells for one positive-sized pack.")
    order = np.argsort(values, axis=-1, kind="stable")
    return order[..., -count:].reshape(*values.shape[:-1], count // pack_size, pack_size)


def score(values, packs):
    """Evaluate one fixed allocation on a lifetime vector or scenario matrix."""
    values = np.asarray(values, dtype=float)
    packs = np.asarray(packs)
    if packs.ndim != 2 or packs.size == 0 or not np.issubdtype(packs.dtype, np.integer):
        raise ValueError("Packs must be a nonempty integer matrix.")
    if len(np.unique(packs)) != packs.size or packs.min() < 0 or packs.max() >= values.shape[-1]:
        raise ValueError("Each selected cell must occur once and be in bounds.")
    return values[..., packs].min(axis=-1).sum(axis=-1)


def posterior(x, y, model):
    """Gaussian posterior for Q(c)=a+b*c/100 using only provided observations."""
    window = model['window_observations']
    x, y = np.asarray(x[-window:]), np.asarray(y[-window:])
    h = np.column_stack([np.ones(len(x)), x / 100.0])
    prior_mean = np.asarray(model['prior_mean'])
    prior_precision = np.diag(1 / np.square(model['prior_sd']))
    noise_var = model['noise_Ah'] ** 2
    cov = np.linalg.inv(prior_precision + h.T @ h / noise_var)
    mean = cov @ (prior_precision @ prior_mean + h.T @ y / noise_var)
    return mean, cov


def rul_samples(mean, cov, current_cycle, normals, model, threshold):
    """Transform uncertain line coefficients into a bounded RUL distribution.

    Flat/rising sampled lines are assigned the explicit horizon cap. This is an
    exploratory model convention, not a calibrated survival distribution.
    """
    theta = mean + normals @ np.linalg.cholesky(cov).T
    falling = theta[..., 1] < -model['minimum_decay_Ah_per_100_cycles']
    slope = np.where(falling, theta[..., 1], -1.0)
    crossing = 100 * (threshold - theta[..., 0]) / slope
    remaining = np.where(falling, crossing - current_cycle, model['maximum_RUL_cycles'])
    # A line already below threshold has no remaining lifetime in this model.
    current_capacity = theta[..., 0] + theta[..., 1] * current_cycle / 100
    remaining = np.where(current_capacity < threshold, 0, remaining)
    return np.clip(remaining, 0, model['maximum_RUL_cycles'])


@dataclass
class ObservationState:
    """Policy-facing state. Contains only revealed measurements, never hidden EOL."""
    history_x: list
    history_q: list
    next_cycles: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    current_cycles: np.ndarray
    samples: np.ndarray
    normals: np.ndarray
    config: dict

    @classmethod
    def create(cls, initial_x, initial_q, next_cycles, config, seed):
        n = len(initial_q)
        normals = np.random.default_rng(seed + 10000).normal(size=(n, config['posterior_samples'], 2))
        state = cls([list(x) for x in initial_x], [list(q) for q in initial_q],
                    next_cycles.astype(float, copy=True), np.zeros((n, 2)), np.zeros((n, 2, 2)),
                    np.zeros(n), np.zeros((n, config['posterior_samples'])), normals, config)
        for i in range(n):
            state.refit(i)
        return state

    def refit(self, i):
        self.means[i], self.covs[i] = posterior(self.history_x[i], self.history_q[i], self.config['model'])
        self.current_cycles[i] = self.history_x[i][-1]
        self.samples[i] = rul_samples(self.means[i], self.covs[i], self.current_cycles[i],
                                      self.normals[i], self.config['model'], self.config['eol_capacity_Ah'])

    def reveal(self, i, cycle, capacity, next_cycle):
        self.history_x[i].append(float(cycle))
        self.history_q[i].append(float(capacity))
        self.next_cycles[i] = next_cycle
        self.refit(i)


def decision_values(state, eligible, rng):
    """Approximate one-observation lookahead using real RPT measurement protocol.

    Shortlist cells whose predicted RUL affects conditional sorted allocations,
    then fantasize the *next capacity observation* and refit the same model.
    Scores include the 5-cycle age consumed by the selected cell. These are MC
    estimates for a restricted mean-sorting rule, not exact optimal EVOI.
    """
    cfg, model = state.config, state.config['model']
    mean_rul = state.samples.mean(axis=1)
    sd_rul = state.samples.std(axis=1)
    base = allocation(mean_rul, cfg['pack_size'])
    eval_count = cfg['decision_evaluation_samples']
    eval_idx = np.arange(eval_count) % state.samples.shape[1]
    scenarios = state.samples[:, eval_idx].T
    base_value = float(score(scenarios, base).mean())

    # Broad cheap shortlist: vary each cell's RUL, reconstruct packs, and see
    # whether the decision changes. This permits spare and non-bottleneck cells.
    proxy = np.zeros(len(mean_rul))
    baseline_membership = np.full(len(mean_rul), -1)
    baseline_membership[base.ravel()] = np.repeat(np.arange(len(base)), cfg['pack_size'])
    for i in eligible:
        for sign in [-1, 1]:
            varied = mean_rul.copy()
            varied[i] = max(0, mean_rul[i] + sign * sd_rul[i])
            alternative = allocation(varied, cfg['pack_size'])
            membership = np.full(len(mean_rul), -1)
            membership[alternative.ravel()] = np.repeat(np.arange(len(alternative)), cfg['pack_size'])
            # Boundaries, pack membership, and bottleneck value can all matter.
            proxy[i] += np.count_nonzero(membership != baseline_membership)
        proxy[i] *= sd_rul[i]
    k = min(cfg['decision_candidate_limit'], len(eligible))
    candidates = eligible[np.argsort(proxy[eligible], kind='stable')[-k:]]
    values = np.full(len(mean_rul), -np.inf)
    fantasies = cfg['decision_fantasies']
    z_y = rng.normal(size=fantasies)
    eval_z = rng.normal(size=(eval_count, 2))
    decision_z = rng.normal(size=(cfg['posterior_samples'], 2))
    for i in candidates:
        next_cycle = state.next_cycles[i]
        h = np.array([1., next_cycle / 100])
        predictive_mean = h @ state.means[i]
        predictive_sd = np.sqrt(h @ state.covs[i] @ h + model['noise_Ah'] ** 2)
        capacity_draws = predictive_mean + predictive_sd * z_y
        expected_values = []
        for observed_q in capacity_draws:
            mu, cov = posterior([*state.history_x[i], next_cycle],
                                [*state.history_q[i], observed_q], model)
            next_samples = rul_samples(mu, cov, next_cycle, eval_z, model, cfg['eol_capacity_Ah'])
            next_mean = rul_samples(mu, cov, next_cycle, decision_z, model,
                                    cfg['eol_capacity_Ah']).mean()
            updated_mean = mean_rul.copy()
            updated_mean[i] = next_mean
            chosen = allocation(updated_mean, cfg['pack_size'])
            updated_scenarios = scenarios.copy()
            updated_scenarios[:, i] = next_samples
            expected_values.append(float(score(updated_scenarios, chosen).mean()))
        cost = next_cycle - state.current_cycles[i]
        values[i] = (np.mean(expected_values) - base_value) / cost
    return values


def choose(policy, state, eligible, rng):
    if policy == 'random':
        return int(rng.choice(eligible)), None
    if policy == 'uniform':
        counts = np.array([len(state.history_x[i]) for i in eligible])
        return int(eligible[np.argmin(counts)]), None
    if policy == 'uncertainty':
        return int(eligible[np.argmax(state.samples[eligible].std(axis=1))]), None
    if policy == 'decision_value':
        values = decision_values(state, eligible, rng)
        chosen = int(np.argmax(values))
        return chosen, float(values[chosen])
    raise ValueError(f'Unknown policy: {policy}')


def load_data(data_dir, config):
    with np.load(data_dir / 'C_full.npz', allow_pickle=False) as archive:
        capacities = archive['C_full'].copy()
        cycles = archive['cycles'].copy()
        ids = archive['cell_ids'].copy()
        source = str(archive['source'])
    eol = np.load(data_dir / 'T_eol.npy', allow_pickle=False)
    censored = np.load(data_dir / 'censor_mask.npy', allow_pickle=False)
    meta = pd.read_csv(data_dir / 'meta.csv')
    if source != 'rpt' or capacities.shape[0] != 86:
        raise ValueError('This pilot requires the verified 86-cell RPT dataset.')
    if censored.any() or not np.array_equal(ids, meta.cell_id.to_numpy()):
        raise ValueError('Censoring or ID alignment needs an explicit data mapping.')
    if not np.array_equal(eol, meta.t_eol.to_numpy()):
        raise ValueError('EOL array and metadata disagree.')
    reconstructed = np.array([cycles[np.flatnonzero(row < config['eol_capacity_Ah'])[0]]
                              for row in capacities])
    if not np.array_equal(reconstructed, eol):
        raise ValueError('Configured EOL threshold does not reproduce the stored labels.')
    maximum = config['maximum_observations']
    initial = config['initial_observations']
    if not 3 <= initial <= maximum <= capacities.shape[1]:
        raise ValueError('Require 3 <= initial observations <= maximum observations.')
    if not np.isfinite(capacities[:, :maximum]).all():
        raise ValueError('This first pilot needs common complete RPT checkpoints.')
    # Validate pilot safety only; no EOL label is passed to the policy.
    if cycles[maximum - 1] >= eol.min():
        raise ValueError('Pilot inspection horizon must precede all stored EOLs.')
    return capacities, cycles, ids, eol


def checkpoint(state, eol, baseline_oracle, policy, seed, budget, ids):
    mean_rul = state.samples.mean(axis=1)
    chosen = allocation(mean_rul, state.config['pack_size'])
    # Only the evaluator accesses eol. The policy sees neither of these arrays.
    actual_rul = np.maximum(0, eol - state.current_cycles)
    oracle = allocation(actual_rul, state.config['pack_size'])
    actual_utility = float(score(actual_rul, chosen))
    oracle_utility = float(score(actual_rul, oracle))
    capacity_packs = allocation(np.array([q[-1] for q in state.history_q]), state.config['pack_size'])
    capacity_utility = float(score(actual_rul, capacity_packs))
    return {
        'policy': policy, 'seed': seed, 'additional_cell_cycles': budget,
        'initial_cell_cycles': int(sum(x[state.config['initial_observations'] - 1] for x in state.history_x)),
        'total_cell_cycles': int(state.current_cycles.sum()),
        'utility_pack_cycles': actual_utility, 'oracle_current_pack_cycles': oracle_utility,
        'performance_score_100': 100 * actual_utility / oracle_utility,
        'relative_regret': 1 - actual_utility / oracle_utility,
        'initial_oracle_retention_percent': 100 * actual_utility / baseline_oracle,
        'capacity_sort_utility_pack_cycles': capacity_utility,
        'capacity_sort_score_100': 100 * capacity_utility / oracle_utility,
        'prediction_MAE_cycles': float(np.mean(np.abs(mean_rul - actual_rul))),
        'prediction_interval_80_coverage': float(np.mean(
            (actual_rul >= np.quantile(state.samples, .1, axis=1)) &
            (actual_rul <= np.quantile(state.samples, .9, axis=1)))),
        'n_selected_cells': int(chosen.size), 'n_spares': int(len(ids) - chosen.size),
        'min_observations': min(map(len, state.history_x)),
        'max_observations': max(map(len, state.history_x)),
        'assignment_cell_ids': json.dumps(ids[chosen].tolist()),
        'spare_cell_ids': json.dumps(ids[np.setdiff1d(np.arange(len(ids)), chosen.ravel())].tolist()),
    }


def run_policy(capacities, cycles, ids, eol, config, policy, seed):
    n = len(ids)
    initial = config['initial_observations']
    maximum = config['maximum_observations']
    rng = np.random.default_rng(seed)
    state = ObservationState.create(np.tile(cycles[:initial], (n, 1)), capacities[:, :initial],
                                    np.full(n, cycles[initial]), config, seed)
    true_initial = np.maximum(0, eol - cycles[initial - 1])
    baseline_oracle = float(score(true_initial, allocation(true_initial, config['pack_size'])))
    requested = sorted(config['additional_budgets_cell_cycles'])
    if not requested or requested[0] != 0:
        raise ValueError('Start the additional budget grid at zero.')
    if not np.all(np.diff(cycles[:maximum]) == 5) or any(b % 5 for b in requested):
        raise ValueError('This pilot budgets actions in actual 5-cycle RPT increments.')
    rows = [checkpoint(state, eol, baseline_oracle, policy, seed, 0, ids)]
    initial_total = int(n * cycles[initial - 1])
    rows[0]['initial_cell_cycles'] = initial_total
    trace = []
    spent = 0
    next_report = 1
    while spent < requested[-1]:
        eligible = np.flatnonzero(np.array([len(v) < maximum for v in state.history_x]))
        if not len(eligible):
            raise ValueError('Budget exceeds available pilot observations.')
        chosen, acquisition = choose(policy, state, eligible, rng)
        column = len(state.history_x[chosen])
        cycle = cycles[column]
        cost = int(cycle - state.current_cycles[chosen])
        # The simulator reveals exactly one newly requested RPT measurement.
        new_capacity = capacities[chosen, column]
        next_cycle = cycles[column + 1] if column + 1 < maximum else np.nan
        state.reveal(chosen, cycle, new_capacity, next_cycle)
        spent += cost
        trace.append({'policy': policy, 'seed': seed, 'step': len(trace) + 1,
                      'cell_id': int(ids[chosen]), 'revealed_cycle': int(cycle),
                      'revealed_capacity_Ah': float(new_capacity), 'cost_cell_cycles': cost,
                      'cumulative_additional_cell_cycles': spent, 'acquisition_value': acquisition})
        if next_report < len(requested) and spent == requested[next_report]:
            row = checkpoint(state, eol, baseline_oracle, policy, seed, spent, ids)
            row['initial_cell_cycles'] = initial_total
            rows.append(row)
            next_report += 1
    return rows, trace


def plot_results(frame, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'uniform': 'Uniform', 'random': 'Random', 'uncertainty': 'Uncertainty',
              'decision_value': 'Decision-value (approx.)'}
    colors = dict(zip(labels, ['#6c7a89', '#b07d34', '#2b8cbe', '#b53b50']))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    for policy, part in frame.groupby('policy', sort=False):
        g = part.groupby('additional_cell_cycles')
        x = np.array(sorted(part.additional_cell_cycles.unique()))
        for ax, key in zip(axes, ['performance_score_100', 'utility_pack_cycles', 'prediction_MAE_cycles']):
            mean, sd = g[key].mean().reindex(x), g[key].std(ddof=0).reindex(x)
            ax.plot(x, mean, marker='o', label=labels[policy], color=colors[policy])
            ax.fill_between(x, mean - sd, mean + sd, color=colors[policy], alpha=.12)
    for ax in axes:
        ax.set_xlabel('Additional diagnostic effort (cell-cycles)')
        ax.grid(alpha=.2)
    initial = frame[frame.additional_cell_cycles == 0].iloc[0]
    axes[0].scatter([0], [initial.capacity_sort_score_100], color='#27845b', marker='D', s=45,
                    label='Capacity-only, no extra test', zorder=5)
    axes[1].scatter([0], [initial.capacity_sort_utility_pack_cycles], color='#27845b', marker='D', s=45,
                    label='Capacity-only, no extra test', zorder=5)
    axes[0].set(ylabel='Pack quality / current-state oracle (%)', title='Allocation quality')
    axes[0].axhline(95, color='black', linestyle='--', alpha=.5, label='95-point target')
    axes[1].set(ylabel='Sum of actual pack RUL (pack-cycles)', title='Deployment value (inspection aging included)')
    axes[2].set(ylabel='Mean absolute RUL error (cycles)', title='Prediction error')
    axes[0].legend(fontsize=8)
    fig.suptitle('REAL 86-CELL PILOT | 10 packs x 8 cells + 6 spares | repeated seeds on one dataset')
    fig.savefig(output / 'real86_comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/real86.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    data_dir = Path(config['data_dir']).expanduser()
    data_dir = (data_dir if data_dir.is_absolute() else ROOT / data_dir).resolve()
    capacities, cycles, ids, eol = load_data(data_dir, config)
    output = ROOT / 'results' / config['run_name']
    output.mkdir(parents=True, exist_ok=True)
    try:
        portable_data_dir = data_dir.relative_to(ROOT.parent).as_posix()
    except ValueError:
        portable_data_dir = data_dir.name
    manifest = {'config': config, 'data_directory': portable_data_dir, 'n_cells': len(ids),
                'n_packs': len(ids) // config['pack_size'], 'n_spares': len(ids) % config['pack_size'],
                'initial_cycle': int(cycles[config['initial_observations'] - 1]),
                'maximum_inspection_cycle': int(cycles[config['maximum_observations'] - 1]),
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in data_dir.iterdir() if p.name in
                                 ['C_full.npz', 'T_eol.npy', 'censor_mask.npy', 'meta.csv', 'cap_raw.parquet']},
                'model_status': 'Untrained local Bayesian linear extrapolation; uncertainty is uncalibrated.',
                'decision_policy_status': 'Restricted-candidate Monte Carlo one-RPT lookahead; not exact EVOI.',
                'repeats_status': 'Seeds repeat the same 86 cells; no independent dataset confidence claim.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    all_rows, all_trace = [], []
    started = time.monotonic()
    for seed in config['seeds']:
        for policy in config['policies']:
            rows, trace = run_policy(capacities, cycles, ids, eol, config, policy, seed)
            all_rows.extend(rows)
            all_trace.extend(trace)
            pd.DataFrame(all_rows).to_csv(output / 'metrics.csv', index=False)
            pd.DataFrame(all_trace).to_csv(output / 'inspection_trace.csv', index=False)
            print(f"seed={seed} {policy}: {rows[-1]['performance_score_100']:.2f}/100, "
                  f"U={rows[-1]['utility_pack_cycles']:.0f}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    frame = pd.DataFrame(all_rows)
    summary = frame.groupby(['policy', 'additional_cell_cycles']).agg(
        score_mean=('performance_score_100', 'mean'), score_sd=('performance_score_100', 'std'),
        utility_mean=('utility_pack_cycles', 'mean'), utility_sd=('utility_pack_cycles', 'std'),
        mae_mean=('prediction_MAE_cycles', 'mean'), coverage80_mean=('prediction_interval_80_coverage', 'mean'),
        capacity_sort_score_mean=('capacity_sort_score_100', 'mean'),
        initial_oracle_retention_mean=('initial_oracle_retention_percent', 'mean')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    plot_results(frame, output)
    target_rows = []
    for policy, part in summary.groupby('policy'):
        reached = part[part.score_mean >= 95].sort_values('additional_cell_cycles')
        target_rows.append({'policy': policy, 'target': 95,
                            'first_observed_budget_mean_score_ge95': None if reached.empty else int(reached.iloc[0].additional_cell_cycles),
                            'note': 'Checkpoint mean only; not a deployable stopping rule or a guarantee.'})
    (output / 'target_status.json').write_text(json.dumps(target_rows, indent=2) + '\n', encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print(f'OUTPUT: {output.relative_to(ROOT).as_posix()}', flush=True)


if __name__ == '__main__':
    main()
