#!/usr/bin/env python3
# coding: utf-8

from myphoebe import *

class Myphoebe2(Myphoebe):
  '''
  A linear model; w. a lot of stuff inherited from Myphoebe.

  Reference: see bayes.pdf.

  '''

  def __init__(self):
    self.x, self.yobs, self.yerr = np.loadtxt("xy.dat", usecols=[0,1,2], unpack=True)
    self.ysyn = None
    self.chi = None

  def model(self, theta):
    a, b, f = theta
    return a*self.x + b

  def lnlike(self, theta):
    '''
    f .. nuisance parameters for underestimated uncertainties (yerr)

    '''
    a, b, f = theta
    s2 = (f*self.yerr)**2
    self.ysyn = self.model(theta)
    self.chi = (self.yobs - self.ysyn)**2/s2
    return -0.5*np.sum(self.chi + np.log(s2) + np.log(2.0*np.pi))

  def lnprior(self, theta):
    a, b, f = theta
    if 0.1 < a < 2.0 and 0.0 < b < 2.0 and 0.5 < f < 5.0:
        return 0.0
    else:
        return -np.inf

  def initial_parameters(self):
    a = 1.0
    b = 0.0
    f = 1.0
    theta = [a, b, f]
    return theta

def main():
  '''
  Test MCMC on a linear model.

  '''

  myphoebe = Myphoebe2()

  run_mcmc(myphoebe, nwalkers=6)

if __name__ == "__main__":
  main()


