#!/usr/bin/env python3
""" Module that contain class Poisson """

class Poisson:
    """ The class that modelate the Poisson distribution
        Args:
             data - list of the data to be used to estimate the distribution
             lambtha - expected number of occurences in a given time frame
    """

    def __init__(self, data=None, lambtha=0.0):
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
            self.lambtha = sum(data) / len(data)
