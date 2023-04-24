#!/bin/bash
timestamp=$(date +%s)
#results_dir="./update/outputs/k_clusters/stackex/"$timestamp
results_dir="./update/outputs/k_clusters/reuters/"$timestamp
data_dir="C:/Users/jcotn/OneDrive/Desktop/XMTC/resources/Reuters"

for i in 2 4 8 32 64 128
do
  for j in 2 4 6 8 10 12 14 16 18 20
  do
    python -u update/train.py -supervision 1 -results_dir $results_dir -data_dir $data_dir -k_lambdas $j -seed $i

  done
done
