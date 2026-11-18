"""common/samples.py 테스트. 실행: python -m pytest tests/test_samples.py -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import samples as SM  # noqa: E402


@pytest.fixture
def fake_dir(tmp_path, monkeypatch):
    N, B, S, T = 5, 7, 4, 390
    Q = np.zeros((N, B, S, T), np.float32)
    for i in range(N):
        for b in range(B):
            Q[i, b] = 100 * i + b            # 값으로 (셀, 진단량 칸)을 알아볼 수 있게
    np.savez(tmp_path / "posterior_samples_r0.npz", Q=Q, t_eol=np.zeros((N, B, S)), cell_ids=np.arange(1, N + 1),
             cycles=1 + 5 * np.arange(T), budgets=np.arange(10, 45, 5), fold=np.zeros(N, int))
    monkeypatch.setenv("SSBT_SAMPLES_DIR", str(tmp_path))
    SM._CACHE.clear()
    return tmp_path


def test_picks_budget_per_cell(fake_dir):
    n = np.array([10, 40, 25, 10, 15])
    s = SM.load_samples(0, n)
    assert s.Q.shape == (5, 4, 390)
    assert [int(s.Q[i, 0, 0]) for i in range(5)] == [0, 106, 203, 300, 401]


def test_rejects_bad_budget_and_length(fake_dir):
    with pytest.raises(ValueError):
        SM.load_samples(0, np.array([0, 10, 10, 10, 10]))
    with pytest.raises(ValueError):
        SM.load_samples(0, np.array([12, 10, 10, 10, 10]))
    with pytest.raises(ValueError):
        SM.load_samples(0, np.full(4, 10))


def test_missing_file(fake_dir):
    with pytest.raises(FileNotFoundError):
        SM.load_samples(3, np.full(5, 10))
