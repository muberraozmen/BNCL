#!/bin/bash
results_dir="./update/outputs/reuters/glove/"
for i in 1 2 3
do
  for j in 0 2 4 8 16
  do
    python3.8 -u update/train.py -supervision $i -seed $j -results_dir $results_dir
  done
done
