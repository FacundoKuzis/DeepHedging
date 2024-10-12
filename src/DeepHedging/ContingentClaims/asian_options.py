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

import tensorflow as tf
from DeepHedging.ContingentClaims import ContingentClaim

class aAsianGeometricCall(ContingentClaim):
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

        # Calculate the product of the prices along the time steps
        product_prices = tf.reduce_prod(paths, axis=1)  # Shape: (num_paths,)

        # Compute the geometric average using the product
        geometric_average_price = tf.pow(product_prices, 1.0 / N)

        # The payoff is max(geometric_average_price - K, 0) for an Asian geometric call option
        payoff = tf.maximum(geometric_average_price - self.strike, 0) * self.amount
        return payoff
    

class AsianGeometricCall2(ContingentClaim):
    """
    A class representing an Asian geometric average call option without log transformation.

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
        Calculates the payoff of the Asian geometric average call option without log transformation.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian geometric average call option.
                              Shape is (num_paths,).
        """
        # Number of time steps (N)
        N = tf.cast(tf.shape(paths)[1], tf.float32)

        # Calculate the product of the prices along the time steps
        product_prices = tf.reduce_prod(paths, axis=1)  # Shape: (num_paths,)

        # Compute the geometric average using the product
        geometric_average_price = tf.pow(product_prices, 1.0 / N)

        # The payoff is max(geometric_average_price - K, 0) for an Asian geometric call option
        payoff = tf.maximum(geometric_average_price - self.strike, 0) * self.amount
        return payoff


class FloatingStrikeAsianGeometricCall(ContingentClaim):
    """
    A class representing an Asian geometric average call option with payoff based on final price minus geometric average.

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
        Calculates the payoff of the Asian geometric average call option as S_T - geometric_average_price.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian geometric average call option.
                              Shape is (num_paths,).
        """
        # Number of time steps (N)
        N = tf.cast(tf.shape(paths)[1], tf.float32)

        # Calculate the product of the prices along the time steps
        product_prices = tf.reduce_prod(paths, axis=1)  # Shape: (num_paths,)

        # Compute the geometric average using the product
        geometric_average_price = tf.pow(product_prices, 1.0 / N)

        # Extract the final asset price S_T from each path
        final_price = paths[:, -1]  # Shape: (num_paths,)

        # The payoff is max(S_T - geometric_average_price, 0)
        payoff = tf.maximum(final_price - geometric_average_price, 0) * self.amount
        return payoff


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
        # Number of time steps (N)
        N = tf.cast(tf.shape(paths)[1], tf.float32)

        # Calculate the arithmetic average of the prices along the time steps
        arithmetic_average = tf.reduce_mean(paths, axis=1)  # Shape: (num_paths,)

        # The payoff is max(A - K, 0) for an Asian arithmetic call option
        payoff = tf.maximum(arithmetic_average - self.strike, 0) * self.amount
        return payoff


class FloatingStrikeAsianCall(ContingentClaim):
    """
    A class representing a Floating Strike Asian call option.

    In this option, the strike price is the arithmetic average of the asset prices over the observation period.

    Arguments:
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the Floating Strike Asian call option.
    """

    def __init__(self, strike, amount=1.0):
        super().__init__(amount)
        self.option_type = 'call'
        self.strike = strike

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the Floating Strike Asian call option.

        Payoff: max(S_T - A, 0), where S_T is the final asset price and A is the arithmetic average.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Floating Strike Asian call option.
                              Shape is (num_paths,).
        """
        # Calculate the arithmetic average of the prices along the time steps
        arithmetic_average = tf.reduce_mean(paths, axis=1)  # Shape: (num_paths,)

        # Extract the final asset price S_T from each path
        final_price = paths[:, -1]  # Shape: (num_paths,)

        # The payoff is max(S_T - A, 0)
        payoff = tf.maximum(final_price - arithmetic_average, 0) * self.amount
        return payoff


class AsianExponentialAverageCall(ContingentClaim):
    """
    A class representing an Asian call option with an exponentially weighted average.

    This option uses an exponential moving average to give more weight to recent asset prices.

    Arguments:
    - strike (float): The strike price of the option.
    - decay_factor (float): The decay factor for exponential weighting (0 < decay_factor <= 1).
    - amount (float, optional): The amount of the option. Default is 1.0.

    Methods:
    - calculate_payoff(self, paths): Calculates the payoff of the Asian exponential average call option.
    """

    def __init__(self, strike, decay_factor=0.5, amount=1.0):
        super().__init__(amount)
        self.strike = strike
        self.decay_factor = decay_factor
        self.option_type = 'call'

    def calculate_payoff(self, paths):
        """
        Calculates the payoff of the Asian exponential average call option.

        Payoff: max(A_exp - K, 0), where A_exp is the exponentially weighted average and K is the strike price.

        Arguments:
        - paths (tf.Tensor): Tensor containing the simulated paths of the underlying asset.
                             Shape is (num_paths, N+1).

        Returns:
        - payoff (tf.Tensor): Tensor containing the payoff of the Asian exponential average call option.
                              Shape is (num_paths,).
        """
        # Number of time steps (N)
        N = tf.cast(tf.shape(paths)[1], tf.float32)

        # Create weights for exponential moving average
        # Weights decay exponentially: w_i = decay_factor^(N - i)
        weights = tf.pow(self.decay_factor, tf.cast(tf.range(tf.shape(paths)[1]), tf.float32))
        weights = weights / tf.reduce_sum(weights)  # Normalize weights

        # Expand weights to match the paths tensor
        weights = tf.expand_dims(weights, axis=0)  # Shape: (1, N+1)

        # Calculate the exponentially weighted average
        exponential_average = tf.reduce_sum(paths * weights, axis=1)  # Shape: (num_paths,)

        # The payoff is max(A_exp - K, 0) for an Asian exponential average call option
        payoff = tf.maximum(exponential_average - self.strike, 0) * self.amount
        return payoff
