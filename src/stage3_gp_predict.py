"""
Stage 3 — 확률적 궤적 예측 모델 (Gaussian Process, 1차 선택)

experiment-design.md Stage 3 명세. 예산 n_i 사이클까지의 특징량을 입력으로,
각 셀의 미래 용량 궤적을 사후분포(샘플 S개)로 예측한다.

GPyTorch 기반 스켈레톤. 실제 커널/학습 루프는 TODO로 남겨둔다.
"""

import numpy as np
import torch
import gpytorch

from config import PROCESSED_DIR


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp(train_x: torch.Tensor, train_y: torch.Tensor, n_iter: int = 200):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def sample_posterior(model, likelihood, test_x: torch.Tensor, n_samples: int = 300):
    """셀 하나에 대해 (n_samples, len(test_x)) 궤적 샘플을 반환."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        samples = pred.rsample(sample_shape=torch.Size([n_samples]))
    return samples.numpy()


def calibration_check(samples: np.ndarray, actual: np.ndarray) -> float:
    """PICP: 90% 예측구간이 실제값을 포함하는 비율."""
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)
    covered = (actual >= lo) & (actual <= hi)
    return float(covered.mean())


if __name__ == "__main__":
    # TODO: features.npz, C_full.npz를 읽어 fold별로 학습·예측·보정 체크를 수행
    print("Stage 3 스켈레톤: fit_gp / sample_posterior / calibration_check 구현 필요")
