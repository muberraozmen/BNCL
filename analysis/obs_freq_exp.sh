#!/bin/bash
timestamp=$(date +%s)
results_dir="./update/outputs/stackex/k_clusters/"$timestamp

data_dir="C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/stackexchange_philosophy/first5000"

for i in 1 2 3
do
  printf 'supervision':$i'\n'
  for j in 2 4 6 8 10 12 14 16 18 20
  do
    python -u update/train.py -supervision $i -results_dir $results_dir -data_dir $data_dir -k_lambdas $j

  done
done
