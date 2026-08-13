#!/bin/bash
apptainer exec --nv env.sif python ~/licenta/_test_asja.py --model Qwen/Qwen2-7B-Instruct --demo --format chatml --modify double

