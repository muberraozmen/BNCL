#!/bin/bash
for i in 5 10 90
do
  for j in 0
  do
    python3.8 -u update/train.py -annotation_ratio $i -supervision 2 -seed $j
  done
done
