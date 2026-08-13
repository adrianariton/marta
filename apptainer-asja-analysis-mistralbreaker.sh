#!/bin/bash
apptainer exec --nv env.sif python ~/licenta/_test_asja_wprompt.py --model GraySwanAI/Mistral-7B-Instruct-RR --demo --format chatml --head 1
apptainer exec --nv env.sif python ~/licenta/_test_asja_wprompt.py --model GraySwanAI/Mistral-7B-Instruct-RR --demo --format chatml --head 2
apptainer exec --nv env.sif python ~/licenta/_test_asja_wprompt.py --model GraySwanAI/Mistral-7B-Instruct-RR --demo --format chatml --head 3
apptainer exec --nv env.sif python ~/licenta/_test_asja_wprompt.py --model GraySwanAI/Mistral-7B-Instruct-RR --demo --format chatml --head 4
apptainer exec --nv env.sif python ~/licenta/_test_asja_wprompt.py --model GraySwanAI/Mistral-7B-Instruct-RR --demo --format chatml --head 5