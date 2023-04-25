#!/bin/bash
data_dir="./update/inputs/reuters/"
results_dir="./update/outputs/reuters/Bert_Bootstraps/"
for i in 1 2 3
do
  for j in 0 2 4 8 32
  do
    python3.8 -u update/train.py -data_dir $data_dir -supervision $i -seed $j -results_dir $results_dir
  done
done
