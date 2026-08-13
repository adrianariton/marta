#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv env2.sif torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    dpo.py
