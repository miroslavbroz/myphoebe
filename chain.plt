#!/usr/bin/gnuplot

set colors classic

set xl "iter"
set yl "theta (vector of free parameters)"
set cbl "walker"

set zeroaxis
set key left
set palette rgbformulae 33,13,10

p \
  "chain.tmp" u 1:3 w p pt 1,\
  "chain.tmp" u 1:4 w p pt 1,\
  "chain.tmp" u 1:5 w p pt 1,\

pa -1

set term png small
set out "chain.png"
rep

