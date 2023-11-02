from enum import Enum
from typing import Iterable

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class Distribution(Enum):
    bernoulli = "bernoulli"
    normal = "normal"


class MSPRTResult:
    def __init__(
        self,
        distribution: Distribution,
        number_of_observations: int,
        sequential_probability_ratio: list[float],
        n_rejection: int,
        decision: str,
        text: str,
        alpha: float,
    ):
        """
        Initializes a mSPRT result which records information about a guiding decision based on
        statistical hypothesis tests.

        Args:
            distribution (Distribution): The distribution used in the mSPRT test.
            number_of_observations (int): The total number of observations processed.
            sequential_probability_ratio (list[float]): The sequential probability ratios for each observation.
            n_rejection (int): The smallest index of observations where the mSPRT test rejects the null hypothesis.
            decision (str): The decision made by the mSPRT test.
            text (str): A string description about the decision and the amount of observations used to reach it.
            alpha (float): The level of significance used in the mSPRT test.
        """
        self.distribution = distribution
        self.number_of_observations = number_of_observations
        self.sequential_probability_ratio = sequential_probability_ratio
        self.n_rejection = n_rejection
        self.decision = decision
        self.text = text
        self.alpha = alpha

    def __str__(self) -> str:
        """
        Returns:
            str: A string representation of the class.
        """

        return (
            f"Distribution: {self.distribution!r},\n"
            + f"number of observations: {self.number_of_observations!r},\n"
            + f"rejection after: {self.n_rejection!r},\n"
            + f"decision: {self.decision!r},\n"
            + f"text: {self.text!r},\n"
            + f"alpha: {self.alpha!r}"
        )

    def plot(self) -> None:
        """
        Plots the sequential probability ratio test results.
        """
        xp = self.sequential_probability_ratio

        y_intercept = 1 / self.alpha

        max_y = np.max([y_intercept, np.max(xp)])
        max_y += 5

        _, ax = plt.subplots()

        ax.set_ylim([0, max_y])
        ax.plot(xp, label="spr")
        ax.axhline(y=y_intercept, color="r", linestyle="--")

        ax.set_xlabel("Observations Collected")
        ax.set_ylabel("Sequential Probability Ratio")

        plt.suptitle("Mixture Sequential Probability Ratio Test")

        if self.n_rejection < len(xp):
            subtitle = f"Null Hypothesis Rejected After {self.n_rejection} Observations"
        else:
            subtitle = "Null Hypothesis Accepted"

        ax.set_title(subtitle, fontsize=8)
        plt.show()


class MSPRT:
    def calculate_mixture_variance(
        self, alpha: float, sigma: float, truncation: float
    ) -> float:
        """
        Calculates the mixture variance.

        Args:
            alpha (float): The level of significance for the mSPRT test.
            sigma (float): The standard deviation used in the calculation of variance.
            truncation (float): The upper limit for the distribution.

        Returns:
            float: The mixture variance.
        """

        if not isinstance(alpha, float) and not (alpha > 0 and alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")

        b = (2 * np.log(1 / alpha)) / np.sqrt(truncation * sigma**2)

        return round(
            sigma**2 * (norm.cdf(-b) / ((1 / b) * norm.pdf(b) - norm.cdf(-b))), 2
        )

    def _validate_inputs(
        self,
        x: Iterable,
        y: Iterable,
        sigma: float,
        tau: float,
        theta: float,
        distribution: Distribution,
        alpha: float,
    ) -> None:
        if x is None or y is None:
            raise ValueError("x and y cannot be empty")
        if len(x) != len(y):
            raise ValueError("x and y must be of same length")
        if distribution == Distribution.normal and not isinstance(sigma, float):
            raise TypeError("sigma must be numeric")
        if not isinstance(theta, float):
            raise TypeError("theta must be numeric")
        if not isinstance(tau, float):
            raise TypeError("tau must be numeric")
        if not tau > 0:
            raise ValueError("tau must be positive")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be numeric")
        if not (alpha > 0 and alpha < 1):
            raise ValueError("alpha value has to be between 0 and 1")

        if distribution not in (
            Distribution.normal.value,
            Distribution.bernoulli.value,
        ):
            raise ValueError("Distribution should be either 'normal' or 'bernoulli'")

    def _calculate_sequential_probability_ratio_for_normal_distribution(
        self,
        x: Iterable,
        y: Iterable,
        warmup_observations: int,
        sigma: float,
        tau: float,
        theta: float,
    ) -> list[float]:
        output = [np.nan] * len(x)

        for i in range(1, len(x) + 1):
            double_variance = 2 * sigma**2
            root_part = np.sqrt(double_variance / (double_variance + (i * tau**2)))
            exponential_part = np.exp(
                (i**2 * tau**2 * (np.mean(x[:i]) - np.mean(y[:i]) - theta) ** 2)
                / (4 * sigma**2 * (2 * sigma**2 + i * tau**2))
            )
            output[i - 1] = root_part * exponential_part

        output[:warmup_observations] = [0] * warmup_observations
        return output

    def _calculate_sequential_probability_ratio_for_bernoulli_distribution(
        self,
        x: Iterable,
        y: Iterable,
        warmup_observations: int,
        tau: float,
        theta: float,
    ) -> list[float]:
        output = [np.nan] * len(x)
        z = x - y

        for i in range(warmup_observations, len(z)):
            Vn = np.mean(x[:i]) * (1 - np.mean(x[:i])) + np.mean(y[:i]) * (
                1 - np.mean(y[:i])
            )
            output[i] = np.sqrt((Vn) / (Vn + i * tau**2)) * np.exp(
                ((i) ** 2 * tau**2 * (np.mean(z[:i]) - theta) ** 2)
                / (2 * Vn * (Vn + i * tau**2))
            )
        output[: warmup_observations + 1] = [0] * (warmup_observations + 1)
        return output

    def calculate_test_statistics(
        self,
        x: Iterable,
        y: Iterable,
        sigma: float,
        tau: float,
        theta: float,
        distribution: Distribution,
        alpha: float,
        warmup_observations: int,
    ) -> MSPRTResult:
        """
        Calculates the test statistics for the mSPRT test.

        Args:
            x, y (Iterable): Two iterable sequences of observations for comparison.
            sigma (float): The standard deviation of the population.
            tau (float): The mixture variance.
            theta (float): The threshold value for the decision rule.
            distribution (Distribution): The distribution type of the data.
            alpha (float): The level of significance for the mSPRT test.
            warmup_observations (int): The number of initial observations disregarded.

        Returns:
            MSPRTResult: The result of the mSPRT test.
        """

        if distribution == Distribution.normal.value:
            output = (
                self._calculate_sequential_probability_ratio_for_normal_distribution(
                    x=x,
                    y=y,
                    warmup_observations=warmup_observations,
                    sigma=sigma,
                    tau=tau,
                    theta=theta,
                )
            )

        elif distribution == Distribution.bernoulli.value:
            output = (
                self._calculate_sequential_probability_ratio_for_bernoulli_distribution(
                    x=x,
                    y=y,
                    warmup_observations=warmup_observations,
                    tau=tau,
                    theta=theta,
                )
            )

        n_rejection = (
            len(x)
            if max(output) <= alpha ** (-1)
            else min(i + 1 for i, v in enumerate(output) if v > alpha ** (-1))
        )
        decision = "Accept H0" if n_rejection >= len(x) else "Accept H1"
        text = f"Decision made after {n_rejection} observations were collected"

        result = MSPRTResult(
            distribution=distribution,
            number_of_observations=len(x),
            sequential_probability_ratio=output,
            n_rejection=n_rejection,
            decision=decision,
            text=text,
            alpha=alpha,
        )

        return result

    def __call__(
        self,
        x: Iterable,
        y: Iterable,
        sigma: float = 0.0,
        theta: float = 0.0,
        truncation: float = 200,
        distribution: Distribution = Distribution.normal.value,
        alpha: float = 0.05,
        warmup_observations: int = 100,
    ) -> MSPRTResult:
        """
        Calls the mSPRT test on the given inputs.

        Args:
            x, y (Iterable): Two iterable sequences of observations for comparison.
            sigma (float, optional): The standard deviation of the population. Defaults to 0.0.
            tau (float, optional): The mixture variance. If not provided, it will be calculated.
            theta (float, optional): The threshold value for the decision rule. Defaults to 0.0.
            truncation (float, optional): The upper limit for the distribution. Defaults to 200.
            distribution (Distribution, optional): The distribution type of the data.
                                                   Defaults to normal distribution.
            alpha (float, optional): The level of significance for the mSPRT test. Defaults to 0.05.
            warmup_observations (int, optional): The number of initial observations disregarded.
                                                 Defaults to 100.

        Returns:
            MSPRTResult: The result of the mSPRT test.
        """
        mixture_variance = self.calculate_mixture_variance(
            alpha=alpha, sigma=sigma, truncation=truncation
        )
        self._validate_inputs(
            x=x,
            y=y,
            sigma=sigma,
            tau=mixture_variance,
            theta=theta,
            distribution=distribution,
            alpha=alpha,
        )
        result = self.calculate_test_statistics(
            x=x,
            y=y,
            sigma=sigma,
            tau=mixture_variance,
            theta=theta,
            distribution=distribution,
            alpha=alpha,
            warmup_observations=warmup_observations,
        )
        return result


msprt = MSPRT()
