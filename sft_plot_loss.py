import json
import matplotlib.pyplot as plt

logs = json.load(open("./datasets/loss_logs.json"))

train_steps = [x["step"] for x in logs["train"]]
train_losses = [x["loss"] for x in logs["train"]]
eval_steps = [x["step"] for x in logs["eval"]]
eval_losses = [x["eval_loss"] for x in logs["eval"]]

plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_losses, label="Train Loss", alpha=0.7)
plt.plot(eval_steps, eval_losses, label="Eval Loss", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Eval Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./train/loss_curve.png", dpi=150)
print("Saved to loss_curve.png")
