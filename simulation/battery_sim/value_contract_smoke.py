"""Reproduce Current-only V under the team contract; no oracle/regret claim."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .value_metrics import (current_only_assignment, effective_future,
                            unused_cells, regrouping_upper_bound, total_value)
from .inventory_protocol import BUDGET_GRID, diagnostic_batch

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'configs/evaluation_contract_v3.json').read_text())
    source = (ROOT / config['data_dir']).resolve() / 'C_full.npz'
    with np.load(source, allow_pickle=False) as z:
        capacities, ids, cycles = z['C_full'], z['cell_ids'], z['cycles']
    if not np.array_equal(cycles, 1 + 5 * np.arange(len(cycles))):
        raise ValueError('The n/5 convention requires RPT columns at cycles 1+5*j.')
    rows = []
    for size in config['pack_sizes']:
        packs = current_only_assignment(capacities, size)
        unused = unused_cells(packs, len(ids))
        rows.append({'pack_size': size, 'N_inventory': len(ids),
                     'n_used': packs.size, 'n_unused': len(unused),
                     'total_diagnostic_cell_cycles': 10 * len(ids),
                     'V_current_only_Ah_cycles': total_value(capacities, 10, packs),
                     'V_O_zero_upper_bound_Ah_cycles': regrouping_upper_bound(effective_future(capacities, 0), size),
                     'unused_ids': json.dumps(ids[unused].tolist())})
    out = ROOT / 'results/value_contract_v3'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'current_only_smoke.csv', index=False)
    n = np.full(len(ids), 10)
    budget_rows, actions = [], []
    for target in BUDGET_GRID:
        rounds = 0
        while n.sum() < target:
            chosen, n = diagnostic_batch(n, np.argsort(n, kind='stable'), target)
            rounds += 1
            actions.append({'checkpoint': target, 'round': rounds,
                            'selected_ids': json.dumps(ids[chosen].tolist()),
                            'batch_size': len(chosen), 'total_cost': int(n.sum())})
        budget_rows.append({'checkpoint': target, 'actual_cost': int(n.sum()),
                            'rounds_in_stage': rounds, 'min_n': int(n.min()), 'max_n': int(n.max())})
    pd.DataFrame(budget_rows).to_csv(out / 'uniform_budget_check.csv', index=False)
    pd.DataFrame(actions).to_csv(out / 'uniform_budget_actions.csv', index=False)
    manifest = {'status': 'current_only_metric_smoke_only_no_oracle_regret_or_GP_run',
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'current_only_n_i': 10, 'observed_RPT_cycles': [1, 6], 'future_first_column': 2,
                'N_basis': config['N_basis'], 'oracle_computation': config['oracle_computation'],
                'total_budget_grid': config['total_budget_grid'],
                'tie_rule': 'stable input cell order',
                'notes': ['Upper bound allows per-step regrouping; it is not a fixed-pack oracle.',
                          'Current-only remains the n=10 baseline; no later observations are used.',
                          'This script does not report total regret until the oracle computation is agreed.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(pd.DataFrame(rows).groupby('pack_size')[['n_used', 'n_unused', 'V_current_only_Ah_cycles']].mean().round(3))


if __name__ == '__main__':
    main()
