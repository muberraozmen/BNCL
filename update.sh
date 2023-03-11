#!/bin/bash
results_dir="./update/outputs/reuters/withoutL2L3/"
for i in 1 2
do
  for j in 0 2 4 8 16 32 64 128 256 512
  do
    python3.8 -u update/train.py -supervision $i -seed $j -results_dir $results_dir
  done
done
