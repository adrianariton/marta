#!/bin/bash
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_orbench_q3next.py 5 0 67 &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_orbench_q3next.py 5 67 134 &
CUDA_VISIBLE_DEVICES=2 apptainer exec --nv env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_orbench_q3next.py 5 134 200 &
wait