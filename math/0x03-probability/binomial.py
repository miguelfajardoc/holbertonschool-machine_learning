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
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = n
                self.p = p
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

    def z_score(self, x):
        """ Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ calculates the value of the PDF for the given x-value
            Args:
                 x - the x-value
        """
        denominator = self.stddev * (2 * Normal.pi) ** (1 / 2)
        exponent = (((x - self.mean) / self.stddev) ** 2) / 2
        PDF = (Normal.e ** (-exponent)) / denominator
        return PDF

    def cdf(self, x):
        """ Calculates the value of the CDF for a given x-value

        """
        erf = Normal.erf((x - self.mean) / (self.stddev * (2 ** (1 / 2))))
        return ((1 + erf) / 2)

    @staticmethod
    def erf(x):
        """ Calculates the aproximate erf function for a given x-value
        """
        firstTerm = 2 / (Normal.pi ** (1 / 2))
        shortSeries = (x - ((x ** 3) / 3) + ((x ** 5) / 10) - ((x ** 7) / 42) +
                       ((x ** 9) / 216))
        return firstTerm * shortSeries
