#!/usr/bin/env python3
# coding: utf-8

from multiprocessing import Pool

from myphoebe import *

def main():

  myphoebe = Myphoebe()

  pool = Pool()
  run_mcmc(myphoebe, pool=pool)

if __name__ == "__main__":
  main()


