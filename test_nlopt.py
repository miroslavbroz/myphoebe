#!/usr/bin/env python3
# coding: utf-8

from myphoebe import *

class Myphoebe2(Myphoebe):
  '''
  A linear model; w. a lot of stuff inherited from Myphoebe.

  Reference: see bayes.pdf.

  '''

  def __init__(self, debug=True):
    self.debug = debug
    self.x, self.yobs, self.yerr = np.loadtxt("xy.dat", usecols=[0,1,2], unpack=True)
    self.ysyn = None
    self.chi = None

  def model(self, theta):
    a, b = theta
    return a*self.x + b

  def initial_parameters(self):
    a = 1.0
    b = 0.0
    theta = [a, b]
    return theta

  def lower_bounds(self):
    a = 0.1
    b = 0.0
    theta = [a, b]
    return theta

  def upper_bounds(self):
    a = 2.0
    b = 2.0
    theta = [a, b]
    return theta

def main():
  '''
  Test MCMC on a linear model.

  '''

  myphoebe = Myphoebe2()

  run_nlopt(myphoebe)

if __name__ == "__main__":
  main()


