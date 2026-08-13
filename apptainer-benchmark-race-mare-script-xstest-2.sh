#!/bin/bash
# apptainer exec --nv env.sif python ~/licenta/benchmark_race_async_batch_bss_mare_script_xstest.py
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/licenta/benchmark_race_async_batch_bss_mare_script_xstest.py 3 &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/licenta/benchmark_race_async_batch_bss_mare_script_xstest.py 5 &
wait