from core.datasets.safety.sorrybench import SorryBench

df = SorryBench().get_first(n=500)
df.to_csv("sorry_bench.csv", index=False)
