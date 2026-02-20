import numpy as np
import tensorflow as tf

from DeepHedging.HedgingInstruments import GBMStock, HestonStock


class MonteCarloPricer:
    """
    Monte Carlo option pricer with finite-difference delta.
    """

    def __init__(self, stock_model, r, T, num_simulations=10_000, seed=None):
        self.stock_model = stock_model
        self.r = float(r)
        self.T = float(T)
        self.num_simulations = int(num_simulations)
        self.seed = None if seed is None else int(seed)

    def _discount_factor(self):
        return float(np.exp(-self.r * self.T))

    def _simulate_gbm_paths_from_dW(self, S0, dW):
        """
        Fast vectorized GBM path generator given Brownian increments dW.
        """
        dt = float(self.stock_model.dt)
        r = float(self.stock_model.r)
        sigma = float(self.stock_model.sigma)
        drift = (r - 0.5 * sigma**2) * dt
        increments = drift + sigma * dW
        log_cum = np.cumsum(increments, axis=1)
        log_full = np.concatenate(
            [np.zeros((log_cum.shape[0], 1), dtype=np.float64), log_cum],
            axis=1,
        )
        paths = float(S0) * np.exp(log_full)
        return tf.convert_to_tensor(paths, dtype=tf.float32)

    def _draw_gbm_dW(self, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        dt = float(self.stock_model.dt)
        return rng.normal(
            loc=0.0,
            scale=np.sqrt(dt),
            size=(self.num_simulations, int(self.stock_model.N)),
        )

    def simulate_paths(self, seed=None, S0=None, dW=None):
        """
        Simulates asset paths using stock_model. For GBM, allows reusing dW across bumps.
        """
        if isinstance(self.stock_model, GBMStock):
            if dW is None:
                dW = self._draw_gbm_dW(seed=seed)
            s0 = float(self.stock_model.S0 if S0 is None else S0)
            return self._simulate_gbm_paths_from_dW(s0, dW)

        # Fallback for non-GBM models.
        original_S0 = getattr(self.stock_model, "S0", None)
        if S0 is not None and original_S0 is not None:
            self.stock_model.S0 = float(S0)
        try:
            draw_seed = self.seed if seed is None else int(seed)
            if isinstance(self.stock_model, HestonStock):
                S_paths, _ = self.stock_model.generate_paths(
                    num_paths=self.num_simulations,
                    random_seed=draw_seed,
                )
                return S_paths
            return self.stock_model.generate_paths(
                num_paths=self.num_simulations,
                random_seed=draw_seed,
            )
        finally:
            if S0 is not None and original_S0 is not None:
                self.stock_model.S0 = original_S0

    def price(self, contingent_claim, paths=None):
        if paths is None:
            paths = self.simulate_paths()
        payoffs = contingent_claim.calculate_payoff(paths)
        payoffs = tf.cast(payoffs, tf.float32)
        discounted_payoff = self._discount_factor() * tf.reduce_mean(payoffs)
        return float(discounted_payoff.numpy())

    def delta(self, contingent_claim, bump_size=0.01, use_common_random_numbers=True, seed=None):
        base_S0 = float(self.stock_model.S0)
        epsilon = max(abs(base_S0) * float(bump_size), 1e-8)
        return self.delta_with_S0(
            contingent_claim=contingent_claim,
            S0=base_S0,
            bump_size=bump_size,
            use_common_random_numbers=use_common_random_numbers,
            seed=seed,
            epsilon=epsilon,
        )

    def delta_with_S0(
        self,
        contingent_claim,
        S0,
        bump_size=0.01,
        use_common_random_numbers=True,
        seed=None,
        epsilon=None,
    ):
        s0 = float(S0)
        eps = max(abs(s0) * float(bump_size), 1e-8) if epsilon is None else float(epsilon)
        up_s0 = s0 + eps
        down_s0 = max(s0 - eps, 1e-8)

        if isinstance(self.stock_model, GBMStock):
            if use_common_random_numbers:
                dW = self._draw_gbm_dW(seed=seed)
                paths_up = self.simulate_paths(S0=up_s0, dW=dW)
                paths_down = self.simulate_paths(S0=down_s0, dW=dW)
            else:
                paths_up = self.simulate_paths(S0=up_s0, seed=seed)
                down_seed = None if seed is None else int(seed) + 1_000_003
                paths_down = self.simulate_paths(S0=down_s0, seed=down_seed)
            price_up = self.price(contingent_claim, paths=paths_up)
            price_down = self.price(contingent_claim, paths=paths_down)
            return float((price_up - price_down) / (2.0 * eps))

        # Fallback for non-GBM models.
        price_up = self.price_with_S0(contingent_claim, up_s0)
        price_down = self.price_with_S0(contingent_claim, down_s0)
        return float((price_up - price_down) / (2.0 * eps))

    def price_with_S0(self, contingent_claim, S0, paths=None):
        if paths is None:
            paths = self.simulate_paths(S0=float(S0))
        payoffs = contingent_claim.calculate_payoff(paths)
        payoffs = tf.cast(payoffs, tf.float32)
        discounted_payoff = self._discount_factor() * tf.reduce_mean(payoffs)
        return float(discounted_payoff.numpy())
