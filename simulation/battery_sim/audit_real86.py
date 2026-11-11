"""Independently reconstruct all reported deployment scores from inspection logs."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .real86 import allocation, score

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'configs/real86.json').read_text())
    out = ROOT / 'results' / config['run_name']
    manifest = json.loads((out / 'manifest.json').read_text())
    data = Path(manifest['data_directory']).expanduser()
    data = (data if data.is_absolute() else ROOT.parent / data).resolve()
    for name, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    with np.load(data / 'C_full.npz') as archive:
        c, cycles, ids = archive['C_full'], archive['cycles'], archive['cell_ids']
    eol = np.load(data / 'T_eol.npy')
    position = {int(cid): i for i, cid in enumerate(ids)}
    metrics = pd.read_csv(out / 'metrics.csv')
    trace = pd.read_csv(out / 'inspection_trace.csv')
    for (policy, seed), rows in metrics.groupby(['policy', 'seed']):
        actions = trace[(trace.policy == policy) & (trace.seed == seed)].sort_values('step')
        counts = np.full(len(ids), config['initial_observations'])
        current = np.full(len(ids), manifest['initial_cycle'], dtype=float)
        spent = 0
        reconstructed = {0: (current.copy(), counts.copy())}
        for action in actions.itertuples():
            i = position[action.cell_id]
            column = counts[i]
            assert action.revealed_cycle == cycles[column]
            assert np.isclose(action.revealed_capacity_Ah, c[i, column], rtol=0, atol=1e-12)
            assert action.cost_cell_cycles == cycles[column] - current[i]
            spent += action.cost_cell_cycles
            assert action.cumulative_additional_cell_cycles == spent
            counts[i] += 1
            current[i] = cycles[column]
            if spent in config['additional_budgets_cell_cycles']:
                reconstructed[spent] = (current.copy(), counts.copy())
        for row in rows.itertuples():
            current, counts = reconstructed[row.additional_cell_cycles]
            truth = np.maximum(0, eol - current)
            packs = np.array([[position[cid] for cid in pack] for pack in json.loads(row.assignment_cell_ids)])
            spares = {position[cid] for cid in json.loads(row.spare_cell_ids)}
            assert packs.shape == (10, 8) and len(set(packs.ravel())) == 80
            assert spares == set(range(86)) - set(packs.ravel())
            actual = float(score(truth, packs))
            oracle = float(score(truth, allocation(truth, config['pack_size'])))
            assert np.isclose(actual, row.utility_pack_cycles)
            assert np.isclose(oracle, row.oracle_current_pack_cycles)
            assert np.isclose(100 * actual / oracle, row.performance_score_100)
            assert np.isclose(current.sum(), row.total_cell_cycles)
            assert row.initial_cell_cycles == 1806
            assert counts.min() == row.min_observations and counts.max() == row.max_observations
            assert row.performance_score_100 <= 100 + 1e-9
    report = {'status': 'passed', 'verified_metric_rows': len(metrics),
              'verified_inspection_actions': len(trace),
              'verified_policy_seed_runs': len(metrics.groupby(['policy', 'seed'])),
              'checks': ['source files unchanged by SHA-256', 'sequential next-RPT revelation',
                         'revealed measurements equal source records', 'diagnostic cost accumulation',
                         '80 unique selected cells plus 6 spares', 'actual post-inspection RUL',
                         'oracle and score reconstruction', 'initial cost and observation counts']}
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
