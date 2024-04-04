#!/bin/bash

device_id=7

export HIP_VISIBLE_DEVICES=${device_id}
output_directory=results_gbench_$(date +"%Y-%m-%d")

mkdir ${output_directory}

for b in $(ls *_BENCH);
do 
        ./${b} --benchmark_out_format=csv --benchmark_out=${output_directory}/${b}.csv >> ${output_directory}.log
done
#nvbench has special argument to set the device id
unset HIP_VISIBLE_DEVICES


output_directory=results_nvbench_$(date +"%Y-%m-%d")

mkdir ${output_directory}

for b in $(ls *_NVBENCH);
do 
        ./${b} -d ${device_id} --csv ${output_directory}/${b}.csv >> ${output_directory}.log
done
