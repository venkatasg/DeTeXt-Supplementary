#!/bin/sh
latex -interaction=batchmode tech.tex
dvisvgm --exact --zoom=-1 -n tech.dvi

