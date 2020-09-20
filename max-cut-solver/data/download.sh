#!/bin/bash

base_url=http://web.stanford.edu/~yyye/yyye/Gset/G


for i in `seq 1 67`; do
    wget ${base_url}${i}
done

for i in 70 72 77 81; do
    wget ${base_url}${i}
done