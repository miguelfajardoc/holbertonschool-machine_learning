#!/usr/bin/env python3
""" Module that contain class Normal """


class Normal:
    """ The class that modelate the Normal distribution
        Args:
             data - list of the data to be used to estimate the distribution
             mean - the mean of the distribution
             stddev - is the standard deviation of the distribution
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """ init normal function
        """
        if data is None:
            if stddev > 0:
                self.mean = float(mean)
                self.stddev = float(stddev)
            else:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = Normal.CalcStdDev(self.mean, data)

    @staticmethod
    def CalcStdDev(mean, data):
        """ Calculate the standar desviation from a given mean and data
        """
        sumatory = 0
        for x in data:
            sumatory += (x - mean) ** 2
        return (sumatory / len(data)) ** (1 / 2)

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

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of “successes”

        """
        if not isinstance(k, int):
            try:
                k = int(k)
            except Exception:
                return 0
        if k < 0:
            return 0
        CDF = 1 - Exponential.e ** (-self.lambtha * k)
        return CDF
