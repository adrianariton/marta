#!/bin/bash
# apptainer exec --nv env.sif python ~/licenta/benchmark_race_async_batch_bss_mare_script_xstest.py
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/ADI/licenta/benchmark_race_async_batch_bss_mare_script_adv.py 1 &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/ADI/licenta/benchmark_race_async_batch_bss_mare_script_adv.py 2 &
CUDA_VISIBLE_DEVICES=2 apptainer exec --nv env.sif python ~/ADI/licenta/benchmark_race_async_batch_bss_mare_script_adv.py 3 &
wait

CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/ADI/licenta/benchmark_race_async_batch_bss_mare_script_adv.py 4 &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/ADI/licenta/benchmark_race_async_batch_bss_mare_script_adv.py 5 &
wait