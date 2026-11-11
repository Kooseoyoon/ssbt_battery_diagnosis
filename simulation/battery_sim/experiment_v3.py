"""V3 design entry point: input preflight and real-capacity controls.

No learned Stage2 result is fabricated when feature files or provenance are absent.
Run from simulation: python -m battery_sim.experiment_v3 --stage preflight|controls
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .shared_folds import validate_folds
from .value_metrics import observed_prefix, total_value, unused_cells

ROOT = Path(__file__).resolve().parents[1]


def capacity_features_v3(cycles, prefixes):
    """V2 ten-feature schema, with early/recent windows of min(3, observed RPTs).

    With two RPTs all three slopes coincide; slope-change/residual are zero.
    This explicit V3 adaptation is not a reproduction of the old E2-B run.
    """
    x, q = np.asarray(cycles, float), np.asarray(prefixes, float)
    if q.ndim != 2 or q.shape[1] != len(x) or len(x) < 2 or not np.isfinite(q).all():
        raise ValueError('At least two finite revealed RPTs per cell required.')
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError('RPT cycles must increase strictly.')
    def slope(a,b):
        centered = a - a.mean()
        return b @ centered / (centered @ centered)
    w = min(3,len(x))
    whole, early, recent = slope(x,q), slope(x[:w],q[:,:w]), slope(x[-w:],q[:,-w:])
    fitted = q.mean(axis=1)[:,None] + whole[:,None]*(x-x.mean())
    return np.column_stack([np.full(len(q),x[-1]),q[:,0],q[:,-1],q[:,0]-q[:,-1],
        whole,early,recent,recent-early,np.sqrt(np.mean((q-fitted)**2,axis=1)),q[:,-1]-q[:,-2]])


def rank_assignment(point, pack_size):
    point = np.asarray(point,float)
    if point.ndim != 1 or not np.isfinite(point).all() or not 1 <= pack_size <= len(point):
        raise ValueError('Expected one finite ranking value per inventory cell.')
    count = len(point)//pack_size*pack_size
    return np.argsort(-point,kind='stable')[:count].reshape(-1,pack_size)


def align_feature_cube(values, feature_ids, diagnostic_cycles, inventory_ids):
    """Proposed adapter: align explicitly by physical IDs; never guess array axes."""
    x, ids, grid = np.asarray(values,float),np.asarray(feature_ids),np.asarray(diagnostic_cycles)
    target = np.asarray(inventory_ids)
    if x.ndim != 3 or x.shape[:2] != (len(ids),len(grid)):
        raise ValueError('Expected explicit cell,diagnostic,feature axes.')
    if len(np.unique(ids)) != len(ids) or set(ids) != set(target):
        raise ValueError('Feature IDs must uniquely cover the full inventory.')
    if not np.array_equal(grid,np.arange(10,41,5)) or np.isinf(x).any():
        raise ValueError('Require diagnostic grid 10..40 and no infinite features.')
    order = {int(cid):j for j,cid in enumerate(ids)}
    return x[[order[int(cid)] for cid in target]]


def input_path(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def file_info(path):
    path=Path(path)
    if not path.is_file():
        display_path = portable_path(path)
        return {'path':display_path,'exists':False}
    return {'path':portable_path(path),'exists':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def portable_path(path):
    path = Path(path).expanduser().resolve()
    try:
        return path.relative_to(ROOT.parent).as_posix()
    except ValueError:
        return path.name


def preflight(config, out):
    data=input_path(config['data_dir'])
    paths={'capacity':data/'C_full.npz','eol':data/'T_eol.npy',
           'features':input_path(config['features_path']),'selection':input_path(config['selection_path']),
           'folds':ROOT/config['folds_path']}
    report={'files':{k:file_info(p) for k,p in paths.items()},'issues':[],
            'learned_comparison_ready':False,'provenance_status':'not_yet_verified',
            'feature_schema_status':'proposed_keys_not_verified_on_team_file',
            'plan':config}
    for k,info in report['files'].items():
        if not info['exists']: report['issues'].append(f'Missing input: {k}')
    try:
        with np.load(paths['capacity'],allow_pickle=False) as z:
            c,ids,cycles=z['C_full'],z['cell_ids'],z['cycles']
        if len(ids)!=86 or len(set(ids))!=86 or c.shape[0]!=86:
            raise ValueError('Expected exactly 86 unique inventory cells.')
        if not np.array_equal(cycles,1+5*np.arange(c.shape[1])):
            raise ValueError('RPT columns must correspond to cycles 1+5*j.')
        folds=validate_folds(pd.read_csv(paths['folds']),ids)
        report['fold_sizes']=folds.groupby('fold').size().to_dict()
        eol=np.load(paths['eol'],allow_pickle=False)
        if eol.shape!=(86,) or not np.isfinite(eol).all() or np.any(eol<=41):
            raise ValueError('Expected finite 86-cell EOL beyond all diagnostic horizons.')
        # Test all policy-visible prefixes, without exposing future to a policy.
        for n in config['diagnostic_grid_per_cell']:
            observed_prefix(c,n)
        report['controls_ready']=True
        if paths['features'].is_file():
            with np.load(paths['features'],allow_pickle=False) as z:
                report['observed_feature_keys']=z.files
                keys=config['feature_adapter_proposed_keys']
                required=[keys[k] for k in ['values','cell_ids','diagnostic_cycles','names']]
                if not all(k in z.files for k in required):
                    raise ValueError('Feature keys differ from proposed adapter; map actual keys explicitly.')
                x=align_feature_cube(z[keys['values']],z[keys['cell_ids']],z[keys['diagnostic_cycles']],ids)
                names=z[keys['names']].tolist()
                if x.shape!=(86,7,148) or len(names)!=148 or len(set(names))!=148:
                    raise ValueError('Expected 148 uniquely named Stage2 features.')
                report['features_shape']=list(x.shape)
                report['missing_feature_fraction']=float(np.isnan(x).mean())
                if paths['selection'].is_file():
                    selected=json.loads(paths['selection'].read_text())[keys['selection_field']]
                    # Initial proposed selection schema: list of feature names.
                    if not isinstance(selected,list) or len(selected)!=20 or len(set(selected))!=20 or not set(selected)<=set(names):
                        raise ValueError('Screened selection must map to 20 unique names; inspect actual schema if different.')
                    report['screened_names']=selected
                    report['feature_schema_status']='schema_valid_provenance_still_unverified'
    except (ValueError,KeyError,TypeError,OSError) as error:
        report['issues'].append(str(error))
    report.setdefault('controls_ready',False)
    report['issues'].append('Before learned comparison: verify prefix provenance and feature selection scope; implement nested fit runner.')
    (out/'readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def controls(config,out):
    with np.load(input_path(config['data_dir'])/'C_full.npz',allow_pickle=False) as z:
        c,ids=z['C_full'],z['cell_ids']
    rows,assignments=[],[]
    for n in config['diagnostic_grid_per_cell']:
        revealed=np.stack(observed_prefix(c,n))
        for m in config['pack_sizes']:
            packs=rank_assignment(revealed[:,-1],m)
            value=total_value(c,n,packs)
            row={'method':'capacity_updated','diagnostic_cycles_per_cell':n,
                 'total_cost':n*len(ids),'N_inventory':len(ids),'pack_size':m,
                 'n_packs':len(packs),'n_unused':len(ids)-packs.size,'V_Ah_cycles':value,
                 'status':'real_capacity_control_only_no_learned_feature_comparison'}
            rows.append(row)
            assignments.append({**row,'pack_cell_ids':ids[packs].tolist(),
                                'unused_cell_ids':ids[unused_cells(packs,len(ids))].tolist()})
            if n==10:
                rows.append({**row,'method':'current_only'})
                assignments.append({**assignments[-1],'method':'current_only'})
    baseline={r['pack_size']:r['V_Ah_cycles'] for r in rows if r['method']=='current_only'}
    for r in rows: r['V_delta_vs_current_only']=r['V_Ah_cycles']-baseline[r['pack_size']]
    pd.DataFrame(rows).to_csv(out/'control_metrics.csv',index=False)
    (out/'control_assignments.json').write_text(json.dumps(assignments,ensure_ascii=False,indent=2)+'\n')
    print('Real-capacity controls complete; learned Stage2 comparison has not run.')
    print(pd.DataFrame(rows).query("method == 'capacity_updated'").pivot(index='total_cost',columns='pack_size',values='V_Ah_cycles').round(3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiment_plan_v3.json')
    parser.add_argument('--stage',choices=['preflight','controls'],default='preflight')
    args=parser.parse_args()
    path=Path(args.config)
    config=json.loads((path if path.is_absolute() else ROOT/path).read_text())
    out=ROOT/config['result_dir'];out.mkdir(parents=True,exist_ok=True)
    report=preflight(config,out)
    print('Control inputs ready:',report['controls_ready'])
    print('Learned comparison ready:',report['learned_comparison_ready'])
    for issue in report['issues']: print(' -',issue)
    if args.stage=='controls':
        if not report['controls_ready']: raise SystemExit('Control inputs failed validation; see readiness.json.')
        controls(config,out)

if __name__=='__main__':
    main()
