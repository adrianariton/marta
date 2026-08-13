apptainer pull ollama.sif docker://ollama/ollama:latest 
for ollama
apptainer pull vllm-ngc.si
f docker://nvcr.io/nvidia/vllm:26.02-py3
for vllm




###
in human labels am in plots: ./plots/judge_ranking

```
cd human_labels
python3 analyze_compare_all.py AnduQVB100.jsonl AdiQVB100.jsonl MORAAIQVB100.jsonl --judge1 "J1" --judge2 "J2"
```


###

pt other judge eval holding, in base folder:

python3 marta_read.py     

si pt a continua evaluarea
python3 marta_manyeval.py       
 si alea de langa ele