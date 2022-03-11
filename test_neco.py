#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt

from myphoebe import *

class Myphoebe2(Myphoebe):
  '''
  A minor modification of Myphoebe (inherited from).

  '''

  def plot_forward_model(self, output='forward_model.png', dpi=300):
    '''
    Plot model from Phoebe2 variables.

    '''

    params = {
      'text.usetex' : False,
      'font.size' : 14,
      'font.family' : 'lmodern',
      }
    plt.rcParams.update(params)

    fig, (ax1, ax2) = plt.subplots(figsize = (10,10), nrows=2, ncols=1)

    s=24
    lw=0.3
    ax1.scatter(self.b['times@lcB@latest@model'].value, self.b['fluxes@lcB@latest@model'].value, color='grey', marker="+", s=s, lw=lw)
    ax1.scatter(self.b['times@lcR@latest@model'].value, self.b['fluxes@lcR@latest@model'].value, color='grey', marker="+", s=s, lw=lw)
    ax1.plot(self.b['times@lcB@latest@model'].value, self.b['fluxes@lcB@latest@model'].value, label='BRITE blue - forward model', color='blue', lw=lw)
    ax1.plot(self.b['times@lcR@latest@model'].value, self.b['fluxes@lcR@latest@model'].value, label='BRITE red - forward model', color='red', lw=lw)

    ax1.set_xlabel(r'$t$ [JD]', labelpad=10)
    ax1.set_ylabel(r'$F$ [1]', labelpad=15)
    ax1.legend(loc='lower left', ncol=2, fontsize=11)
    ax1.set_xlim(0,5.73245103)

    ax2.scatter(self.b['times@primary@rv1@latest@model'].value, self.b['rvs@primary@rv1@latest@model'].value, color='grey', marker="+", s=s, lw=lw)
    ax2.scatter(self.b['times@secondary@rv2@latest@model'].value, self.b['rvs@secondary@rv2@latest@model'].value, color='grey', marker="+", s=s, lw=lw)
    ax2.plot(self.b['times@primary@rv1@latest@model'].value, self.b['rvs@primary@rv1@latest@model'].value, color='cyan', label='RV primary - forward model', lw=lw)
    ax2.plot(self.b['times@secondary@rv2@latest@model'].value, self.b['rvs@secondary@rv2@latest@model'].value, color='orange', label='RV secondary - forward model', lw=lw)

    ax2.plot([0,6], [0,0], c='grey', lw=lw)
    ax2.set_xlabel(r'$t$ [d]', labelpad=10)
    ax2.set_ylabel(r'RV [km/s]', labelpad=10)
    ax2.legend(loc='upper right', ncol=2, fontsize=11)
    ax2.set_xlim(0,5.73245103)
    ax2.set_ylim(-350,350)

    plt.tight_layout()
    plt.savefig(output, dpi=dpi)


def main():
  '''
  Main program.

  '''

  myphoebe = Myphoebe2()

  theta = initial_parameters()

  myphoebe.model(theta)
  myphoebe.plot_forward_model()

if __name__ == "__main__":
  main()


