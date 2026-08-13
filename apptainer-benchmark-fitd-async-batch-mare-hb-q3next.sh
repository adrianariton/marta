#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 1 &
#CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 2 &
#CUDA_VISIBLE_DEVICES=2 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 3 &
#wait

#CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 4 &
#CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 5 &
#CUDA_VISIBLE_DEVICES=2 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 6 &
#wait


CUDA_VISIBLE_DEVICES=0 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 7 &
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv env.sif python ~/licenta/benchmark_fitd_async_batch_bss_mare_script_harmbench_q3next.py 8 &
wait
