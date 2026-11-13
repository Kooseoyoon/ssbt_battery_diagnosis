import unittest

import numpy as np
import torch

from battery_sim.constitutive_pinn import constitutive_rate, fit_constitutive_model
from battery_sim.physics_experiment import early_features


class ConstitutivePINNChecks(unittest.TestCase):
    def test_sqrt_linear_rate_matches_equation(self):
        rate = constitutive_rate(
            "sqrt_linear",
            torch.tensor([4.0]),
            torch.tensor([[0.2, 0.01]]),
            epsilon=0.0,
        )
        self.assertAlmostEqual(float(rate[0]), 0.06, places=6)

    def test_exponential_linear_rate_starts_at_initial_decay_rate(self):
        rate = constitutive_rate(
            "exponential_linear",
            torch.tensor([0.0]),
            torch.tensor([[0.4, 2.0, 0.01]]),
            epsilon=0.05,
        )
        self.assertAlmostEqual(float(rate[0]), 0.21, places=6)

    def test_unknown_equation_fails_explicitly(self):
        with self.assertRaises(ValueError):
            constitutive_rate("unknown", torch.tensor([1.0]), torch.ones((1, 2)), 0.05)

    def test_both_pinn_equations_train_and_predict_finite_coefficients(self):
        cycles = 1 + 5 * np.arange(8)
        capacities = np.tile([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, np.nan], (3, 1))
        eol = np.full(3, 31)
        config = {
            "trajectory_landmarks_per_cell": 3,
            "eol_capacity_Ah": 1.44,
            "maximum_predicted_rul_cycles": 2000,
            "minimum_trajectory_scale_Ah": 0.02,
            "hidden_units": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "data_huber_scale_Ah": 0.05,
            "collocation_points_per_cell": 3,
            "physics_residual_scale_Ah_per_RPT": 0.02,
            "sqrt_epsilon_RPT": 0.05,
            "sqrt_k_s_scale": 0.05,
            "linear_k_scale": 0.002,
            "exponential_k_scale": 0.2,
            "exponential_tau_scale": 50.0,
            "minimum_tau_RPT": 0.1,
            "epochs": 2,
        }
        for equation in ("sqrt_linear", "exponential_linear"):
            with self.subTest(equation=equation):
                predict, _ = fit_constitutive_model(
                    equation,
                    0.1,
                    capacities,
                    cycles,
                    eol,
                    np.arange(3),
                    early_features(capacities),
                    config,
                    seed=10,
                )
                forecast, parameters = predict(
                    early_features(capacities)[0], [11, 16, 21]
                )
                self.assertTrue(np.isfinite(forecast).all())
                self.assertTrue(np.isfinite(parameters).all())
                self.assertTrue((parameters > 0).all())


if __name__ == "__main__":
    unittest.main()
