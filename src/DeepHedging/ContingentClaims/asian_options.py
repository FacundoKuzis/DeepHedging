import tensorflow as tf
from DeepHedging.ContingentClaims import ContingentClaim

class AsianArithmeticCall(ContingentClaim):
    """
    A class representing an Asian arithmetic average call option.

    Arguments:
    - strike (float): The strike price of the option.
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the Asian arithmetic average call option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.option_type = 'call'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the Asian arithmetic average call option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian arithmetic average call option.
                              Shape is (num_paths,).
        """
        # Calculate the arithmetic average of the underlying asset prices over time
        average_price = tf.reduce_mean(paths, axis=1)  # Shape: (num_paths,)

        payoff = tf.maximum(average_price - self.strike, 0) * self.amount
        return payoff

class AsianArithmeticPut(ContingentClaim):
    """
    A class representing an Asian arithmetic average put option.

    Arguments:
    - strike (float): The strike price of the option.
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the Asian arithmetic average put option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.option_type = 'put'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the Asian arithmetic average put option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian arithmetic average put option.
                              Shape is (num_paths,).
        """
        # Calculate the arithmetic average of the underlying asset prices over time
        average_price = tf.reduce_mean(paths, axis=1)  # Shape: (num_paths,)

        payoff = tf.maximum(self.strike - average_price, 0) * self.amount
        return payoff

class AsianGeometricCall(ContingentClaim):
    """
    A class representing an Asian geometric average call option.

    Arguments:
    - strike (float): The strike price of the option.
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the Asian geometric average call option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.option_type = 'call'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the Asian geometric average call option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian geometric average call option.
                              Shape is (num_paths,).
        """
        # Number of time steps
        N = tf.cast(tf.shape(paths)[1], tf.float32)

        # Calculate the sum of the log prices
        log_paths = tf.math.log(paths)
        sum_log_prices = tf.reduce_sum(log_paths, axis=1)  # Shape: (num_paths,)

        # Compute the geometric average using logs
        geometric_average_price = tf.exp(sum_log_prices / N)

        # The payoff is max(geometric_average_price - K, 0) for an Asian geometric call option
        payoff = tf.maximum(geometric_average_price - self.strike, 0) * self.amount
        return payoff

class AsianGeometricPut(ContingentClaim):
    """
    A class representing an Asian geometric average put option.

    Arguments:
    - strike (float): The strike price of the option.
        - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the Asian geometric average put option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.option_type = 'put'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the Asian geometric average put option.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian geometric average put option.
                              Shape is (num_paths,).
        """
        # Number of time steps
        N = tf.cast(tf.shape(paths)[1], tf.float32)

        # Calculate the sum of the log prices
        log_paths = tf.math.log(paths)
        sum_log_prices = tf.reduce_sum(log_paths, axis=1)  # Shape: (num_paths,)

        # Compute the geometric average using logs
        geometric_average_price = tf.exp(sum_log_prices / N)

        # The payoff is max(K - geometric_average_price, 0) for an Asian geometric put option
        payoff = tf.maximum(self.strike - geometric_average_price, 0) * self.amount
        return payoff
