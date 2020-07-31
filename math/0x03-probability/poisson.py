#!/usr/bin/env python3
""" Module that contain class Poisson """


class Poisson:
    """ The class that modelate the Poisson distribution
        Args:
             data - list of the data to be used to estimate the distribution
             lambtha - expected number of occurences in a given time frame
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ init lambtha function """
        if data is None:
            if lambtha >= 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                print("{} not list".format(data))
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))


    def pmf(self, k):
        """ calculates the value of the PMF for the given number of successes
            Args:
                 k - the number of "successes"
        """
        if not isinstance(k, int):
            try:
                k = int(k)
            except:
                return 0
        PMF = self.lambtha ** k / (Poisson.factorial(k) * Poisson.e ** self.lambtha)
        return PMF

    @staticmethod
    def factorial(n):
        """ calculates the factorial of a number """

        if n == 0 or n == 1:
            return 1
        else:
            return n * Poisson.factorial(n-1)
