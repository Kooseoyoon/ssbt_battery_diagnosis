"""Cell-level fold manifest shared by feature and trajectory-model stages."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def create_folds(cell_ids, seed=42):
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or len(ids) < 5 or len(np.unique(ids)) != len(ids):
        raise ValueError('Require at least five unique physical cell IDs.')
    assignment = np.full(len(ids), -1, dtype=int)
    for fold, (_, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(ids)):
        assignment[test] = fold
    return pd.DataFrame({'cell_id': ids, 'fold': assignment}).sort_values('cell_id').reset_index(drop=True)


def validate_folds(frame, cell_ids):
    if not {'cell_id', 'fold'} <= set(frame.columns):
        raise ValueError('Fold CSV must contain cell_id and fold.')
    if frame[['cell_id', 'fold']].isna().any().any() or frame.cell_id.duplicated().any():
        raise ValueError('Each physical cell must occur exactly once, without missing values.')
    if set(frame.cell_id) != set(cell_ids):
        raise ValueError('Fold IDs do not match the dataset.')
    if set(frame.fold) != set(range(5)) or not np.all(frame.fold.to_numpy() % 1 == 0):
        raise ValueError('Use fold indices 0..4.')
    return frame[['cell_id', 'fold']].sort_values('cell_id').reset_index(drop=True)
