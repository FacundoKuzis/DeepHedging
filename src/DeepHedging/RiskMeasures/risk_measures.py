import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class RiskMeasure:
    """
    The base class for risk measures that take a PnL (Profit and Loss) tensor and return a corresponding risk measure.

    Methods:
    - calculate(self, pnl): Abstract method that must be implemented by subclasses to calculate the risk measure.
    """

    def calculate(self, pnl):
        """
        Abstract method that must be implemented by subclasses to calculate the risk measure.

        Arguments:
        - pnl (tf.Tensor): Tensor containing the PnL values.

        Returns:
        None. (Must be implemented in subclasses.)
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
