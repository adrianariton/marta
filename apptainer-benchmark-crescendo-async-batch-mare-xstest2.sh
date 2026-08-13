#!/bin/bash
export HF_TOKEN="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"
apptainer exec --nv --env HF_TOKEN=$HF_TOKEN  env.sif python ~/licenta/benchmark_crescendo_async_batch_bss_mare_script_xstest2.py

