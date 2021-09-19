#!/bin/bash



obj="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
mvtec_ad_data_path="../data/mvtec_anomaly_detection"

for i in $obj;
do 
    echo ${i}
    python train.py --obj  ${i} --data_path ${mvtec_ad_data_path}
done 
