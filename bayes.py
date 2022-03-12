#!/usr/bin/env python

"""
bayes.py
Aproximace primkou, vc. vypoctu nejistot (Monte Carlo Markov Chain; MCMC)

"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import emcee
import corner

def minus_ln_likelihood(params, x, y, sigma):
    """Verohodnost -ln p(x_i,y_i,sigma_i|a,b,f)"""
    a, b, f = params
    model = a*x + b
    s2 = (f*sigma)**2
    J = 0.5*np.sum((y-model)**2/s2 + np.log(s2) + np.log(2.0*np.pi))
    print "J = ", J
    return J

def minus_ln_prior(params):
    """Prior -ln p(a,b,f)"""
    a, b, f = params
    if 0.1 < a < 2.0 and 0.0 < b < 2.0 and 0.5 < f < 5.0:
        return 0.0
    else:
        return -np.inf

def ln_posterior(params, x, y, sigma):
    """Posterior ln p(a,b,f|x_i,y_i,sigma_i)"""
    tmp = minus_ln_prior(params)
    if not np.isfinite(tmp):
        return -np.inf
    else:
        return -(minus_ln_prior(params) + minus_ln_likelihood(params, x, y, sigma))

def main():
    """Nacteni dat, maximalizace verohodnosti, vypocet MCMC, rohovy obrazek, ..."""
    x, y, sigma = np.loadtxt("xy.dat", usecols=[0,1,2], unpack=True)

    a = 1.0
    b = 0.0
    f = 1.0
    result = scipy.optimize.minimize(lambda *args: minus_ln_likelihood(*args), \
        [a,b,f], args=(x,y,sigma), method='Nelder-Mead', tol=1.0e-4)
    a, b, f = result.x
    print result

    ndim = 3
    walkers = 32
    position = [ result.x + 1.0e-4*np.random.rand(ndim) for i in range(walkers) ]
    sampler = emcee.EnsembleSampler(walkers, ndim, ln_posterior, args=(x,y,sigma))
    sampler.run_mcmc(position, 500)

    fig = plt.figure()
    plt.subplot(311)
    plt.plot(sampler.chain[:,:,0].transpose())
    plt.xlabel("krok")
    plt.ylabel("$a$")
    plt.subplot(312)
    plt.plot(sampler.chain[:,:,1].transpose())
    plt.ylabel("$b$")
    plt.subplot(313)
    plt.plot(sampler.chain[:,:,2].transpose())
    plt.ylabel("$f$")
    plt.savefig("chain.png")

    samples = sampler.chain[:,100:,:].reshape((-1,ndim))

    fig = corner.corner(samples, labels=["$a$","$b$","$f$"], truths=[a,b,f])
    fig.savefig("corner.png")

    fig = plt.figure()
    for a_tmp, b_tmp, f in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(x, a_tmp*x + b_tmp, color='k', alpha=0.1)
    plt.plot(x, a*x + b, color='r')
    plt.errorbar(x, y, yerr=sigma, fmt='k+', ecolor='y', capsize=2.0)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("xy.png")

    a, b, f = map(lambda x: (x[1], x[2]-x[1], x[0]-x[1]), \
        zip(*np.percentile(samples, [16,50,84], axis=0)))
    print "a = ", a
    print "b = ", b
    print "f = ", f

if __name__ == "__main__":
    main()

