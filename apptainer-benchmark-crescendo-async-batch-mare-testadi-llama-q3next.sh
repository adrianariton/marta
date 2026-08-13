#!/bin/bash
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_testall_q3next.py 5 0 167 &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_testall_q3next.py 5 167 334 &
CUDA_VISIBLE_DEVICES=2 apptainer exec --nv env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_testall_q3next.py 5 334 500 &
wait