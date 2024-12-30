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
        self.plot_name = {'en': 'Unknown', 'es': 'Desconocido'}

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
    """
    def __init__(self):
        self.name = 'MSE'
        self.plot_name = {'en': 'Mean Squared Error',
                          'es': 'Error cuadrático medio'}

    def calculate(self, pnl):
        """
        Calculates the Mean Squared Error (MSE) of the given PnL.
        """
        mse = tf.reduce_mean(tf.square(pnl))
        return mse

class SMSE(RiskMeasure):
    """
    A class representing the Semi Mean Squared Error (SMSE) risk measure,
    which penalizes only negative PnL.
    """
    def __init__(self):
        self.name = 'SMSE'
        self.plot_name = {'en': 'Semi Mean Squared Error',
                          'es': 'Error cuadrático medio semipositivo'}

    def calculate(self, pnl):
        """
        Calculates the Semi Mean Squared Error (SMSE) of the given PnL.
        """
        negative_pnl = tf.boolean_mask(pnl, pnl < 0)
        smse = tf.reduce_mean(tf.square(negative_pnl))
        return smse

class MAE(RiskMeasure):
    """
    A class representing the Mean Absolute Error (MAE) risk measure.
    """
    def __init__(self):
        self.name = 'MAE'
        self.plot_name = {'en': 'Mean Absolute Error',
                          'es': 'Error absoluto medio'}

    def calculate(self, pnl):
        """
        Calculates the Mean Absolute Error (MAE) of the given PnL.
        """
        mae = tf.reduce_mean(tf.abs(pnl))
        return mae

class VaR(RiskMeasure):
    """
    A class representing the Value at Risk (VaR) risk measure.

    Arguments:
    - alpha (float): The confidence level.
    """
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.name = f'VaR_{int(alpha*100)}'
        self.plot_name = {'en': f'VaR ({int(alpha*100)}%)',
                          'es': f'VaR ({int(alpha*100)}%)'}

    def calculate(self, pnl):
        """
        Calculates the Value at Risk (VaR) of the given PnL.
        """
        var = tfp.stats.percentile(pnl, q=(1 - self.alpha) * 100, interpolation='linear')
        return -var

class CVaR(RiskMeasure):
    """
    A class representing the Conditional Value at Risk (CVaR) risk measure.

    Arguments:
    - alpha (float): The confidence level.
    """
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.name = f'CVaR_{int(alpha*100)}'
        self.plot_name = {'en': f'CVaR ({int(alpha*100)}%)',
                          'es': f'CVaR ({int(alpha*100)}%)'}

    def calculate(self, pnl):
        """
        Calculates the Conditional Value at Risk (CVaR) of the given PnL.
        """
        var = tfp.stats.percentile(pnl, q=(1 - self.alpha) * 100, interpolation='linear')
        cvar = tf.reduce_mean(tf.boolean_mask(pnl, pnl <= var))
        return -cvar

class WorstCase(RiskMeasure):
    """
    A class representing the Worst Case risk measure.
    """
    def __init__(self):
        self.name = 'WorstCase'
        self.plot_name = {'en': 'Worst Case Scenario',
                          'es': 'Peor caso'}

    def calculate(self, pnl):
        """
        Calculates the Worst Case scenario of the given PnL.
        """
        worst_case = tf.reduce_min(pnl)
        return -worst_case

class Entropy(RiskMeasure):
    """
    A class representing the Entropy risk measure.
    """
    def __init__(self):
        self.name = 'Entropy'
        self.plot_name = {'en': 'Entropy',
                          'es': 'Entropía'}

    def calculate(self, pnl):
        """
        Calculates the entropy of the given PnL.
        """
        normalized_pnl = pnl - tf.reduce_min(pnl) + 1e-9
        probabilities = normalized_pnl / tf.reduce_sum(normalized_pnl)
        entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
        return entropy

class Mean(RiskMeasure):
    """
    A class representing the Mean risk measure.
    """
    def __init__(self):
        self.name = 'Mean'
        self.plot_name = {'en': 'Mean',
                          'es': 'Media'}

    def calculate(self, pnl):
        """
        Calculates the Mean of the given PnL.
        """
        mean = tf.reduce_mean(pnl)
        return mean

class StdDev(RiskMeasure):
    """
    A class representing the Standard Deviation risk measure.
    """
    def __init__(self):
        self.name = 'StdDev'
        self.plot_name = {'en': 'Standard Deviation',
                          'es': 'Desvío estándar'}

    def calculate(self, pnl):
        """
        Calculates the Standard Deviation of the given PnL.
        """
        std_dev = tf.math.reduce_std(pnl)
        return std_dev
