#!/usr/bin/gnuplot

set colors classic

set xl "iter"
set yl "chi^2"

set logscale y

p "chi2_func.tmp" u 0:1 w lp

pa -1

set term png small
set out "chi2_func.png"
rep


