#!/bin/bash

noise=0.1
data_list=("cifar10" "cifar100" "svhn")
model_list=("vgg13" "resnet18" "vgg16" "resnet34" "vgg11" "resnet18")
opt="adam"
lr=0.0001

gpu_id=0
for loop in 0 1 2
do
	data=${data_list[$loop]}
	for in_loop in 0 1
	do
		model_index=`expr 2 \* $loop + $in_loop`
		model=${model_list[$model_index]}
		for run_id in 0 1 2 3 4
		do
			echo "CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 -u train_epoch_wise_noise.py $data $model $noise True $opt $lr $run_id > ${loop}_${gpu_id}_${run_id}.log &"
			CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 -u train_epoch_wise_noise.py $data $model $noise True $opt $lr $run_id > ${loop}_${gpu_id}_${run_id}.log &
			gpu_id=`expr $gpu_id + 1`
			gpu_id=`expr $gpu_id % 8`
		done
	done
done
