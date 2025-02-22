#!/bin/bash

Mvtec=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
for item in "${Mvtec[@]}"; do
  echo  "$item"
  python mvtec_ad_evaluation/evaluate_experiment.py --dataset_base_dir '/home/zwk/dataset/mvtec_anomaly_detection' --anomaly_maps_dir './output/cdw3/anomaly_maps/mvtec_ad/' --output_dir './output/cdw3/metrics/mvtec_ad/' --evaluated_objects "$item"
done