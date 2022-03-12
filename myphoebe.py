#!/usr/bin/env python3
# coding: utf-8

__author__ = "Alzbeta Oplitilova (Betsimsim@seznam.cz)"
__version__ = "Mar 11th 2022"

import sys
import time
import numpy as np
import nlopt
import emcee

import phoebe
from phoebe import u


class Myphoebe(object):
  '''
  My wrapper for Phoebe2 computations.

  '''

  def __init__(self, debug=True):
    '''
    Initialisation (input data, Phoebe2, fixed parameters).

    '''
    self.debug = debug

    # Input data
    tb, mb_obs, mb_err = np.loadtxt('blue-ready-to-fit_test.dat', unpack=True, usecols=[0, 1, 2])
    tr, mr_obs, mr_err = np.loadtxt('red-ready-to-fit_test.dat', unpack=True, usecols=[0, 1, 2])
    t1, rv1_obs, rv1_err = np.loadtxt('rv1-ready-to-fit_test.dat', unpack=True, usecols=[0, 1, 2])
    t2, rv2_obs, rv2_err = np.loadtxt('rv2-ready-to-fit_test.dat', unpack=True, usecols=[0, 1, 2])

    fluxb_obs = 10.0**(-0.4*mb_obs)
    fluxr_obs = 10.0**(-0.4*mr_obs)
    fluxb_err = fluxb_obs*np.log10(10.0)*0.4*mb_err  # ERROR!
    fluxr_err = fluxr_obs*np.log10(10.0)*0.4*mr_err  # ERROR!

    # Single vector(s)
    self.x = np.r_[tb, tr, t1, t2]
    self.yobs = np.r_[fluxb_obs, fluxr_obs, rv1_obs, rv2_obs]
    self.yerr = np.r_[fluxb_err, fluxr_err, rv1_err, rv2_err]
    self.ysyn = None
    self.chi = None

    logger = phoebe.logger('error', filename='mylog.log')

    self.b = phoebe.default_binary()

    self.b.add_dataset('lc', compute_times=tb, dataset='lcB', intens_weighting='photon', passband='BRITE:blue')
    self.b.add_dataset('lc', compute_times=tr, dataset='lcR', intens_weighting='photon', passband='BRITE:red')
    self.b.add_dataset('rv', compute_times=t1, dataset='rv1', intens_weighting='photon', passband='Johnson:V')
    self.b.add_dataset('rv', compute_times=t2, dataset='rv2', intens_weighting='photon', passband='Johnson:V')

    self.b['ntriangles@primary']   = 1500
    self.b['ntriangles@secondary'] = 500

    # Fixed parameters
    self.b.set_value('period', component='binary', value=5.73245103*u.d)

    self.b.set_value('l3_mode', 'fraction', dataset='lcB')
    self.b.set_value('l3_mode', 'fraction', dataset='lcR')
    self.b.set_value('l3_frac', 0.273, dataset='lcB')
    self.b.set_value('l3_frac', 0.273, dataset='lcR')

    # Other parameters
    self.b.set_value('atm', component='primary',   value='blackbody')
    self.b.set_value('atm', component='secondary', value='blackbody')

    self.b.set_value(qualifier='ld_mode', dataset='lcB', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='lcR', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv1', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv2', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='lcB', component='secondary', value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='lcR', component='secondary', value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv1', component='secondary', value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv2', component='secondary', value='manual')

    self.b.set_value(qualifier='ld_func',   dataset='lcB', component='primary',   value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='lcR', component='primary',   value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='rv1', component='primary',   value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='rv2', component='primary',   value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='lcB', component='secondary', value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='lcR', component='secondary', value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='rv1', component='secondary', value='quadratic')
    self.b.set_value(qualifier='ld_func',   dataset='rv2', component='secondary', value='quadratic')

    self.b.set_value(qualifier='ld_coeffs', dataset='lcB', component='primary',   value=[0.150834,0.312043])
    self.b.set_value(qualifier='ld_coeffs', dataset='lcR', component='primary',   value=[0.308675,0.258258])
    self.b.set_value(qualifier='ld_coeffs', dataset='rv1', component='primary',   value=[0.347032,0.240089])
    self.b.set_value(qualifier='ld_coeffs', dataset='rv2', component='primary',   value=[0.347032,0.240089])
    self.b.set_value(qualifier='ld_coeffs', dataset='lcB', component='secondary', value=[0.150834,0.312043])  # ERROR!
    self.b.set_value(qualifier='ld_coeffs', dataset='lcR', component='secondary', value=[0.308675,0.258258])  # ERROR!
    self.b.set_value(qualifier='ld_coeffs', dataset='rv1', component='secondary', value=[0.347032,0.240089])  # ERROR!
    self.b.set_value(qualifier='ld_coeffs', dataset='rv2', component='secondary', value=[0.347032,0.240089])  # ERROR!

    self.b.set_value(qualifier='ld_mode_bol', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode_bol', component='secondary', value='manual')

    self.b.set_value('gravb_bol', component='primary',   value=1.0)
    self.b.set_value('gravb_bol', component='secondary', value=1.0)

    self.b.set_value('irrad_frac_refl_bol', component='primary',   value=1.0)
    self.b.set_value('irrad_frac_refl_bol', component='secondary', value=1.0)

    self.b.flip_constraint('mass@primary',   solve_for='sma@binary')
    self.b.flip_constraint('mass@secondary', solve_for='q')

    # Save 'twigs'
    if self.debug:
      f = open('twigs.txt', 'w')
      for twig in self.b.twigs:
        f.write("%s\n" % twig)
      f.close()


  def model(self, theta):
    '''
    Synthetic fluxes from the model.

    :param theta: Vector of free parameters.
    :return:

    '''
    T1,T2,R1,R2,I,S,M1,M2,e,omega,gamma,T0 = theta

    self.b.set_value('ecc',        component='binary',    value=e)
    self.b.set_value('per0',       component='binary',    value=omega*u.deg)
    self.b.set_value('incl',       component='binary',    value=I*u.deg)
    self.b.set_value('t0_supconj', component='binary',    value=T0*u.d)
    self.b.set_value('teff',       component='primary',   value=T1*u.K)
    self.b.set_value('teff',       component='secondary', value=T2*u.K)
    self.b.set_value('requiv',     component='primary',   value=R1*u.solRad)
    self.b.set_value('requiv',     component='secondary', value=R2*u.solRad)
    self.b.set_value('mass',       component='primary',   value=M1*u.solMass)
    self.b.set_value('mass',       component='secondary', value=M2*u.solMass)
    self.b.set_value('vgamma',                            value=gamma*u.km/u.second)

    self.b.run_compute(distortion_method='roche', irrad_method='wilson', ltte=False, rv_method='flux-weighted', rv_grav=False)

    fluxb_syn = self.b['fluxes@lcB@latest@model'].value
    fluxr_syn = self.b['fluxes@lcR@latest@model'].value
    rv1_syn =   self.b['rvs@primary@rv1@latest@model'].value
    rv2_syn =   self.b['rvs@secondary@rv2@latest@model'].value

    # Normalisation
    fluxb_nor = S*fluxb_syn/np.amax(fluxb_syn)
    fluxr_nor = S*fluxr_syn/np.amax(fluxr_syn)

    # Save model
    if self.debug:
      self.b.save('forward_model.phoebe')
      f = open('forward_model.txt', 'w')
      np.savetxt(f, np.column_stack((self.b['times@lcB@latest@model'].value, fluxb_nor)), fmt='%22.16f', header="times,fluxb_nor")
      np.savetxt(f, np.column_stack((self.b['times@lcR@latest@model'].value, fluxr_nor)), fmt='%22.16f', header="times,fluxr_nor")
      np.savetxt(f, np.column_stack((self.b['times@primary@rv1@latest@model'].value, rv1_syn)), fmt='%22.16f', header="times,rv1_syn")
      np.savetxt(f, np.column_stack((self.b['times@secondary@rv2@latest@model'].value, rv2_syn)), fmt='%22.16f', header="times,rv2_syn")
      f.close()

    return np.r_[fluxb_nor, fluxr_nor, rv1_syn, rv2_syn]

  def chi2(self, theta):
    '''
    Computes chi^2.

    :param theta: Vector of free parameters.
    :return:

    '''
    if self.debug:
      print('theta = ', theta)

    self.ysyn = self.model(theta)
    self.chi = ((self.yobs - self.ysyn)/self.yerr)**2
    chi_sum = np.sum(self.chi)

    if self.debug:
      print('chi_sum = ', chi_sum)

      f = open(f'chi2_func.tmp', 'a')
      f.write("%16.8f %d " % (chi_sum, len(self.yobs)))
      for tmp in theta: f.write(" %22.16f" % tmp)
      f.write("\n")
      f.close()

    return chi_sum

  def lnlike(self, theta):
    '''
    Likelihood -ln p(data|theta).

    :param theta: Vector of free parameters.
    :return:

    '''
    self.ysyn = self.model(theta)
    self.chi = ((self.yobs - self.ysyn)/self.yerr)**2
    return -0.5*np.sum(self.chi + np.log(self.yerr**2) + np.log(2.0*np.pi))

  def lnprior(self, theta):
    '''
    Prior -ln p(theta). Uninformative; assures appropriate ranges.

    :param theta: Vector of free parameters.
    :return:

    '''
    T1,T2,R1,R2,I,S,M1,M2,e,omega,gamma,T0 = theta

    if  25000 < T1    < 35000 and \
        20000 < T2    < 30000 and \
        10    < R1    < 20    and \
        2     < R2    < 10    and \
        65    < I     < 89    and \
        0.9   < S     < 1.1   and \
        18    < M1    < 35    and \
        3     < M2    < 20    and \
        0     < e     < 0.2   and \
        90    < omega < 180   and \
        0     < gamma < 35    and \
        -0.52 < T0    < -0.32:
      return 0.0
    else:
      return -np.inf

  def lnprob(self, theta):
    '''
    Posterior ln p(theta|data).

    :param theta: Vector of free parameters.
    :return:

    '''
    lp = self.lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + self.lnlike(theta)

  def initial_parameters():
    '''
    Setting of initial parameters

    :return theta: Vector of free parameters.

    '''
    T1    = 29368.0   # K
    T2    = 25119.0   # K
    R1    = 14.0      # R_Sol
    R2    = 4.1507    # R_Sol
    I     = 77.5175   # deg
    S     = 1.01419   # 1
    M1    = 25.1647   # M_Sol
    M2    = 8.4338    # M_Sol
    e     = 0.08915   # 1
    omega = 157.6877  # deg
    gamma = 17.48     # km/s
    T0    = -0.428699 # d

    theta = T1,T2,R1,R2,I,S,M1,M2,e,omega,gamma,T0
    return theta

  def lower_bounds(self):
    '''
    Lower bounds for nlopt.

    :return theta: Vector of free parameters.

    '''
    T1    = 25000
    T2    = 20000
    R1    = 10
    R2    = 2
    I     = 65
    S     = 0.9
    M1    = 18
    M2    = 3
    e     = 0
    omega = 90
    gamma = 0
    T0    = -0.52

    theta = T1,T2,R1,R2,I,S,M1,M2,e,omega,gamma,T0
    return theta

  def upper_bounds():
    '''
    Upper bounds for nlopt.

    :return theta: Vector of free parameters.

    '''
    T1    = 30000
    T2    = 30000
    R1    = 20
    R2    = 10
    I     = 89
    S     = 1.1
    M1    = 35
    M2    = 20
    e     = 0.2
    omega = 180
    gamma = 35
    T0    = -0.32

    theta = T1,T2,R1,R2,I,S,M1,M2,e,omega,gamma,T0
    return theta


def run_nlopt(myphoebe, algorithm=nlopt.LN_NELDERMEAD, ftol=1e-6, maxeval=100):
  '''
  Run optimisation.

  :param myphoebe: Ref. to myphoebe object.
  :param algorithm: Algorithm, e.g., nlopt.LN_NELDERMEAD, nlopt.LN_SBPLX, ...
  :param ftol: Tolerance to stop.
  :param maxeval: Maximum number of evaluations.
  :return:

  '''

  def myfunc(theta, grad):
    return myphoebe.chi2(theta)

  theta = myphoebe.initial_parameters()

  dim = len(theta)
  opt = nlopt.opt(algorithm, dim)

  print('Number of dimensions:', opt.get_dimension())
  print('Algorithm:', opt.get_algorithm_name())

  opt.set_lower_bounds(myphoebe.lower_bounds())
  opt.set_upper_bounds(myphoebe.upper_bounds())

  opt.set_ftol_rel(ftol)
  opt.set_maxeval(maxeval)
  opt.set_min_objective(myfunc)

  best_fit_theta = opt.optimize(theta)
  best_fit_chi2 = opt.last_optimum_value()

  print('Result code: ', opt.last_optimize_result())

  np.savetxt('best_fit.csv', np.r_[best_fit_chi2,best_fit_theta], delimiter=',', header='best_fit_chi2,best_fit_theta')

  print('run_nlopt() has ended sucessfully!')


def p0_func(theta, nwalkers=None, delta=0.01):
  '''
  Creating initial positions of walkers.

  :param theta: Vector of free parameters.
  :param nwalkers: Number of walkers.
  :param delta: Dispesion of random numbers.
  :return:

  '''
  p0 = []
  for i in range(nwalkers):
    tmp = []
    for j in range(len(theta)):
      tmp.append(np.random.normal(theta[j], delta))
    p0.append(tmp)

  return np.array(p0)


def run_mcmc(myphoebe, nwalkers=25, niter=1000, seed=1, thin=1, **kwarg):
  '''
  Running Monte-Carlo-Markov-Chain method.

  :param myphoebe: Ref. to myphoebe object.
  :param nwalkers: Number of walkers; minimum is 2 times the number of free parameters.
  :param niter: Number of iterations.
  :param seed: Random seed.
  :param thin: Use only every thin step from the chain.
  :return:

  '''
  theta = myphoebe.initial_parameters()
  print('theta = ', theta)

  np.random.seed(seed)
  p0 = p0_func(theta, nwalkers=nwalkers)

  ndim = len(theta)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, myphoebe.lnprob, **kwarg)

  print("Running production...")
  t1 = time.time()

  pos, prob, state = sampler.run_mcmc(p0, 1, progress=True)

  for i in range(niter):
    print('iter = ', i)

    pos, prob, state = sampler.run_mcmc(None, 1, progress=True)

    with open(f'chain.tmp', 'a') as f:
      for j in range(0,len(pos)):
        f.write("%d %d" % (i, j))
        for k in range(0,len(pos[j])):
          f.write(" %22.16f" % (pos[j][k]))
        f.write("\n")

    with open(f'prob.tmp', 'a') as f:
      for j in range(0,len(prob)):
        f.write("%d %d %16.8f\n" % (i, j, prob[j]))

    k = 0
    for tmp in pos.T:
      if k < len(pos[0]):
        k += 1
      with open(f'pos{k}.txt', 'a') as f:
        f.write("\n")
        np.savetxt(f, tmp, fmt='%22.16f', newline=' ', delimiter='')  

  print("Average acceptance fraction:", np.around(np.mean(sampler.acceptance_fraction),3), "(it should be between 0.2-0.5)")
  try:
     print("Autocorrelation time estimate:", sampler.get_autocorr_time(), "(it should be around n x 10)")
     #thin = int(np.mean(sampler.get_autocorr_time())/2.0)
  except:
     print("Warning: Autocorrelation time can not be reliably estimated!")

  samples = sampler.flatchain
  theta_maxprob = samples[np.argmax(sampler.flatlnprobability)]
  chain = sampler.get_chain(thin=thin, flat=True, discard=0)

  np.savetxt('theta_maxprob.csv', theta_maxprob, delimiter=',')
  np.savetxt('chain.csv', chain, delimiter=',')

  t2 = time.time()
  print("Time: ", t2-t1, " s = ", (t2-t1)/3600.0, " h")

  print('run_mcmc() has ended sucessfully!')

def main():
  '''
  Main program.

  '''

  myphoebe = Myphoebe()

  theta = myphoebe.initial_parameters()

#  myphoebe.model(theta)
#  myphoebe.plot_forward_model()

#  run_nlopt(myphoebe)

  run_mcmc(myphoebe)

#  print(vars(myphoebe))  # dbg

if __name__ == "__main__":
  main()


