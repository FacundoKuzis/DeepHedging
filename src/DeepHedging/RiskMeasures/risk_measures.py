import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class RiskMeasure:
    """
    The base class for risk measures that take a PnL (Profit and Loss) tensor and return a corresponding risk measure.

    Methods:
    - calculate(self, pnl): Abstract method that must be implemented by subclasses to calculate the risk measure.
    """
    def __init__(self):
        self.name = 'unknown'

    def calculate(self, pnl):
        """
        Abstract method that must be implemented by subclasses to calculate the risk measure.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, pnl):
        return self.calculate(pnl)
    
class MSE(RiskMeasure):
    """
    A class representing the Mean Squared Error (MSE) risk measure.

    Methods:
    - calculate(self, pnl): Calculates the MSE of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - mse (tf.Tensor): Scalar tensor containing the calculated MSE.
    """
    def __init__(self):
        self.name = 'MSE'

    def calculate(self, pnl):
        """
        Calculates the Mean Squared Error (MSE) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - mse (tf.Tensor): Scalar tensor containing the calculated MSE.
        """
        mse = tf.reduce_mean(tf.square(pnl))
        return mse

class SMSE(RiskMeasure):
    """
    A class representing the Semi Mean Squared Error (SMSE) risk measure,
    which penalizes only negative PnL.

    Methods:
    - calculate(self, pnl): Calculates the SMSE of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - smse (tf.Tensor): Scalar tensor containing the calculated SMSE.
    """
    def __init__(self):
        self.name = 'SMSE'

    def calculate(self, pnl):
        """
        Calculates the Semi Mean Squared Error (SMSE) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - smse (tf.Tensor): Scalar tensor containing the calculated SMSE.
        """
        negative_pnl = tf.boolean_mask(pnl, pnl < 0)
        smse = tf.reduce_mean(tf.square(negative_pnl))
        return smse


class MAE(RiskMeasure):
    """
    A class representing the Mean Absolute Error (MAE) risk measure.

    Methods:
    - calculate(self, pnl): Calculates the MAE of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - mae (tf.Tensor): Scalar tensor containing the calculated MAE.
    """
    def __init__(self):
        self.name = 'MAE'

    def calculate(self, pnl):
        """
        Calculates the Mean Absolute Error (MAE) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - mae (tf.Tensor): Scalar tensor containing the calculated MAE.
        """
        mae = tf.reduce_mean(tf.abs(pnl))
        return mae


class VaR(RiskMeasure):
    """
    A class representing the Value at Risk (VaR) risk measure.

    Arguments:
    - alpha (float): The confidence level (e.g., 0.95 for 95% confidence).

    Methods:
    - calculate(self, pnl): Calculates the VaR of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - var (tf.Tensor): Scalar tensor containing the calculated VaR.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.name = f'VaR_{int(alpha*100)}'

    def calculate(self, pnl):
        """
        Calculates the Value at Risk (VaR) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - var (tf.Tensor): Scalar tensor containing the calculated VaR.
        """
        var = tfp.stats.percentile(pnl, q=(1 - self.alpha) * 100, interpolation='linear')
        return -var

class CVaR(RiskMeasure):
    """
    A class representing the Conditional Value at Risk (CVaR) risk measure.

    Arguments:
    - alpha (float): The confidence level (e.g., 0.95 for 95% confidence).

    Methods:
    - calculate(self, pnl): Calculates the CVaR of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - cvar (tf.Tensor): Scalar tensor containing the calculated CVaR.
    """

    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.name = f'CVaR_{int(alpha*100)}'

    def calculate(self, pnl):
        """
        Calculates the Conditional Value at Risk (CVaR) of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - cvar (tf.Tensor): Scalar tensor containing the calculated CVaR.
        """
        # Compute the Value at Risk (VaR)
        var = tfp.stats.percentile(pnl, q=(1 - self.alpha) * 100, interpolation='linear')

        # Compute CVaR by taking the mean of the losses that are less than or equal to VaR
        cvar = tf.reduce_mean(tf.boolean_mask(pnl, pnl <= var))

        return -cvar

class WorstCase(RiskMeasure):
    """
    A class representing the Worst Case risk measure, 
    which returns the minimum value of the PnL (worst-case scenario).

    Methods:
    - calculate(self, pnl): Calculates the worst-case scenario of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - worst_case (tf.Tensor): Scalar tensor containing the calculated worst-case scenario.
    """
    def __init__(self):
        self.name = 'WorstCase'

    def calculate(self, pnl):
        """
        Calculates the Worst Case scenario of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - worst_case (tf.Tensor): Scalar tensor containing the calculated worst-case scenario.
        """
        worst_case = tf.reduce_min(pnl)
        return -worst_case

class Entropy(RiskMeasure):
    """
    A class representing the Entropy risk measure, 
    which quantifies the uncertainty in the PnL distribution.

    Methods:
    - calculate(self, pnl): Calculates the entropy of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - entropy (tf.Tensor): Scalar tensor containing the calculated entropy.
    """
    def __init__(self):
        self.name = 'Entropy'

    def calculate(self, pnl):
        """
        Calculates the entropy of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - entropy (tf.Tensor): Scalar tensor containing the calculated entropy.
        """
        # Normalize the PnL values to obtain a probability distribution
        normalized_pnl = pnl - tf.reduce_min(pnl) + 1e-9  # Shift to positive values to avoid log(0)
        probabilities = normalized_pnl / tf.reduce_sum(normalized_pnl)

        # Compute the entropy
        entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
        return entropy

class Mean(RiskMeasure):
    """
    A class representing the Mean risk measure.

    Methods:
    - calculate(self, pnl): Calculates the mean of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - mean (tf.Tensor): Scalar tensor containing the calculated mean.
    """
    def __init__(self):
        self.name = 'Mean'

    def calculate(self, pnl):
        """
        Calculates the Mean of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - mean (tf.Tensor): Scalar tensor containing the calculated mean.
        """
        mean = tf.reduce_mean(pnl)
        return mean

class StdDev(RiskMeasure):
    """
    A class representing the Standard Deviation risk measure.

    Methods:
    - calculate(self, pnl): Calculates the standard deviation of the given PnL.
      - Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.
      - Returns:
        - std_dev (tf.Tensor): Scalar tensor containing the calculated standard deviation.
    """
    def __init__(self):
        self.name = 'StdDev'

    def calculate(self, pnl):
        """
        Calculates the Standard Deviation of the given PnL.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        - std_dev (tf.Tensor): Scalar tensor containing the calculated standard deviation.
        """
        std_dev = tf.math.reduce_std(pnl)
        return std_dev
