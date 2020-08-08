#!/usr/bin/env python3
""" Module that contain class Exponential """


class Exponential:
    """ The class that modelate the Exponential distribution
        Args:
             data - list of the data to be used to estimate the distribution
             lambtha - expected number of occurences in a given time frame
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ init exponential function """
        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """ calculates the value of the PDF for the given number of successes
            Args:
                 k - the number of "successes"
        """
        if x < 0:
            return 0
        PMF = self.lambtha * (Exponential.e ** (-self.lambtha * x))
        return PMF

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of “successes”

        """
        if k < 0:
            return 0
        CDF = 1 - Exponential.e ** (-self.lambtha * k)
        return CDF
