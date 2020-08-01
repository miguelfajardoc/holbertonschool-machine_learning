#!/usr/bin/env python3
""" Module that contain class Binomial """


class Binomial:
    """ The class that modelate the binomial distribution
        Args:
             data - list of the data to be used to estimate the distribution
             n - the number of bernoulli trials
             p - is the probability of a “success”
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """ init normal function
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            var = Binomial.CalcVariance(mean, data)
            self.p = 1 - (var / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    @staticmethod
    def CalcVariance(mean, data):
        """ Calculate the standar desviation from a given mean and data
        """
        sumatory = 0
        for x in data:
            sumatory += (x - mean) ** 2
        return sumatory / len(data)

    def pmf(self, k):
        """ calculates the value of the PDF for the given number of successes
            Args:
                 k - number of successes
        """
        if not isinstance(k, int):
            try:
                k = int(k)
            except Exception:
                return 0
        if k < 0:
            return 0
        return (Binomial.combinatory(self.n, k) * (self.p ** k) *
                ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes

        """
        if not isinstance(k, int):
            try:
                k = int(k)
            except Exception:
                return 0
        if k < 0:
            return 0
        for succes in range(k + 1):
            cdf += self.pmf(i)
        return cdf

    @staticmethod
    def combinatory(n, y):
        """ Calculates the combinatory for two given parameters
        """
        return (Binomial.factorial(n) / (Binomial.factorial(n - y) *
                                         Binomial.factorial(y)))

    @staticmethod
    def factorial(n):
        """ calculates the factorial of a number """

        if n <= 1:
            return 1
        else:
            return n * Binomial.factorial(n-1)
