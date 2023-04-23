#!/bin/bash
results_dir="./update/outputs/reuters/glove/"
for i in 1 2 3
do
  for j in 0
  do
    python3.8 -u update/train.py -supervision $i -seed $j -results_dir $results_dir
  done
done
