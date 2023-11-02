# Python mSPRT Package:

This package provides a Python implementation for calculating the Mixture Sequential Probability Ratio Test (mSPRT). 

mSPRT is a statistical hypothesis test that can be used to decide if a observed data supports one of two hypotheses, based on a sequence of independent and identically distributed observations.

Main functionalities:
1. Calculating mixture variance

$$
\tau^2 = \sigma^2 \frac{\Phi(-b)}{\frac{1}{b}\phi(b)-\Phi(-b)}
$$

2. Calculating test statistic for normal distribution

$$
\tilde{\Lambda}_n = \sqrt{\frac{2\sigma^2}{V_n + n\tau^2}}\exp\left(\frac{n^2\tau^2(\bar{Y}_n - \bar{X}_n-\theta_0)^2}{4\sigma^2(2\sigma^2+n\tau^2)}\right).
$$

3. Calculating test statistic for Bernoulli distribution
$$\tilde{\Lambda}_n = \sqrt{\frac{V_n}{V_n + n\tau^2}}\exp{\left(\frac{n^2\tau^2(\bar{Y}_n - \bar{X}_n-\theta_0)^2}{2V_n(V_n+n\tau^2)}\right)}$$



## Installation:

The mSPRT package can be easily installed using pip:

```bash
pip install msprt
```

## Pre-requisite
Python >=3.10;<3.13

## Dependencies:

The mSPRT package depends on the following Python libraries:
- Numpy
- Scipy
- Matplotlib

These dependencies can also be easily installed using pip:

```bash
pip install numpy scipy matplotlib
```

## How to Use:

First, import the mSPRT package:

```python
from msprt import msprt
```

Then, prepare the two sample lists that you want to compare.

```python
x = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
y = [0.2, 0.1, 0.4, 0.6, 0.7, 0.8]
```

Next, call the `msprt` object with observation lists, along with the parameters for the mSPRT test, such as the `alpha` and the `theta` values (by default it assumes you are using a normal distribution and alpha is set to 0.05).

```python
result = msprt(x=x, y=y, sigma=1.0)
```

If you want to use a Bernoulli distribution, specify it as such:

```python
result = msprt(x=x, y=y, theta=0.5, distribution='bernoulli')
```

To plot the results, use the `plot` method:

```python
result.plot()
```

For detailed information about each parameter, please refer to the comments in the source code.

## Contact:

If you find any problems with the implementation, you can leave the ticket on Github.

[mSPRT GitHub Page](https://github.com/ovidijusku/msprt)

## License:

This project is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. See the `LICENSE` file for more information.

## References (real heroes)
1. Johari, R., Pekelis, L., & Walsh, D. J. (2019). Always Valid Inference: Bringing Sequential Analysis to A/B Testing. arXiv:1512.04922 [math.ST]. [Link to the paper](https://doi.org/10.48550/arXiv.1512.04922)
2. The R and C++ implementations of the paper are available in the GitHub repository maintained by Erik Stenberg: [GitHub Repository](https://github.com/erik-stenberg/mixtureSPRT).
