#!/bin/bash
Mvtec=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
for item in "${Mvtec[@]}"; do
  echo  "$item"
  python efficientad.py --dataset mvtec_ad --model_size small --subdataset "$item"
done