#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FULL_STEPS=100000 nohup bash train.sh stage2 8 &