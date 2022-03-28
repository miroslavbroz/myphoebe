#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt

from myphoebe import *

params = {
        'text.usetex' : False,
        'font.size'   : 15,
        'font.family' : 'lmodern',
        }
plt.rcParams.update(params)


def phase(HJD, P=5.732436, T0=2457733.8493):
  phi = ((HJD-T0)/P)%1.
  return phi


class Myphoebe2(Myphoebe):
  '''
  A minor modification of Myphoebe (inherited form).

  '''

  def chi2_contribs(self):
    '''
    Print chi^2 contributions for each dataset.

    '''
    for dts in np.unique(self.dataset):
      ids = np.where(self.dataset==dts)
      chi_ = self.chi[ids]
      ids = np.where(~np.isnan(chi_))
      nans = len(chi_)-len(ids[0])
      print('chi2 = %16.8f, n = %d, dataset = %d, nans = %s' % (np.sum(chi_[ids]), len(ids[0]), dts, nans))

  def plot_comp_phase(self, output='test_comp_phase.png', dpi=300):
    '''
    Plot model from PHOEBE 2 variables.

    '''

    fig, (ax1, ax2) = plt.subplots(figsize = (10,10), nrows=2, ncols=1)

    fig.suptitle(r'$\chi^2$', fontsize=20)

    s  = 26
    ms = 4
    lw = 0.3

    ids = np.where(self.dataset==1)

    ax1.errorbar(phase(self.x[ids]), self.yobs[ids], self.yerr[ids], color='navy', marker=".", ms=ms, lw=lw, fmt='o', label='BRITE blue - observed')
    ax1.scatter(phase(self.x[ids]), self.ysyn[ids], color='black', marker="+", s=s, lw=lw, label='BRITE - sythetic')
    for i in ids:
      ax1.plot([phase(self.x[i]), phase(self.x[i])], [self.yobs[i], self.ysyn[i]], color='red', lw=1)

    max_lcB_F, min_lcB_F = max(np.r_[self.yobs[ids], self.ysyn[ids]]), min(np.r_[self.yobs[ids], self.ysyn[ids]])

    ids = np.where(self.dataset==2)
    tmp = 0.05
    ax1.errorbar(phase(self.x[ids]), self.yobs[ids]+tmp, self.yerr[ids], color='red', marker=".", ms=ms, lw=lw, fmt='o', label='BRITE red - observed')
    ax1.scatter(phase(self.x[ids]), self.ysyn[ids]+tmp, color='black', marker="+", s=s, lw=lw)
    for i in ids:
      ax1.plot([phase(self.x[i]), phase(self.x[i])], [self.yobs[i]+tmp, self.ysyn[i]+tmp], color='orange', lw=1)

    ax1.set_xlabel(r'$\varphi$', labelpad=10)
    ax1.set_ylabel(r'$F$ [1]', labelpad=15)
    ax1.legend(loc='lower left', ncol=3, fontsize=11)

    ax1.set_xlim(-0.02, 1.02)

    max_lcR_F, min_lcR_F = max(np.r_[self.yobs[ids], self.ysyn[ids]]), min(np.r_[self.yobs[ids], self.ysyn[ids]])

    # marigns
    min_F = min(min_lcB_F, min_lcR_F)
    max_F = max(max_lcB_F, max_lcR_F)
#    ax1.set_ylim(min_F-0.15*(max_F-min_F), max_F+0.04*(max_F-min_F))

    ids = np.where(self.dataset==3)
    ax2.errorbar(phase(self.x[ids]), self.yobs[ids], self.yerr[ids], color='green', marker=".", ms=ms, lw=lw, fmt='o', label='RV$_1$ - observed')
    ax2.scatter(phase(self.x[ids]), self.ysyn[ids], color='black', marker="+", s=s, lw=lw, label=r'RV$_1$ - sythetic')
    for i in ids:
      ax2.plot([phase(self.x[i]), phase(self.x[i])], [self.yobs[i], self.ysyn[i]], color='red', lw=1)
     
    ids = np.where(self.dataset==4)
    ax2.errorbar(phase(self.x[ids]), self.yobs[ids], self.yerr[ids], color='purple', marker=".", ms=ms, lw=lw, fmt='o', label='RV$_2$ - observed')
    ax2.scatter(phase(self.x[ids]), self.ysyn[ids], color='gray', marker="+", s=s, lw=lw, label=r'RV$_2$ - synthetic')
    for i in ids:
      ax2.plot([phase(self.x[i]), phase(self.x[i])], [self.yobs[i], self.ysyn[i]], color='orange', lw=1)

    ax2.plot([-0.02, 1.02], [0, 0], c = 'gray', lw=2*lw)

    ax2.set_xlabel(r'$\varphi$', labelpad=10)
    ax2.set_ylabel(r'RV [km/s]', labelpad=10)
    ax2.legend(loc='lower left', ncol=4, fontsize=11)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-390,390)

    plt.tight_layout()
    plt.savefig(output, dpi=dpi)


def main():
  '''
  Main program.

  '''

  myphoebe = Myphoebe2()

#  theta = myphoebe.initial_parameters()
#  print('theta = ', theta)

  f = open("initial_parameters.tmp", "r")
  s = f.readline()
  f.close()
  theta = list(map(lambda x: float(x), s.split()[2:]))
  print('theta = ', theta)

  myphoebe.chi2(theta)
  myphoebe.chi2_contribs()
  myphoebe.plot_comp_phase()
   

if __name__ == "__main__":
  main()


