import matplotlib

matplotlib.use("Qt5Agg")  # IMPORTANT

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# -------------------------
# Load
# -------------------------
path = Path(sys.argv[1])
df = pd.DataFrame(json.loads(l) for l in path.open())


def norm_text(x):
    return "\n".join(x) if isinstance(x, list) else str(x)


df["text"] = df["text"].apply(norm_text)

# -------------------------
# Figure
# -------------------------
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 2])

ax_metrics = fig.add_subplot(gs[:, 0])
ax_timeline = fig.add_subplot(gs[0, 1])
ax_text = fig.add_subplot(gs[1, 1])

colors = {
    "ATTACK": "red",
    "AGENT": "blue",
    "IS_ATTACK_REFUSED": "gray",
    "EVAL": "green",
}

models = df["gen_id"].unique()


# -------------------------
# Simulation
# -------------------------
def simulate_until(frame):
    visible = []
    metrics = []

    for i in range(frame + 1):
        row = df.iloc[i]
        tag = row["tag"]

        if tag == "IS_ATTACK_REFUSED" and row["text"].lower() == "true":
            while visible and visible[-1]["tag"] in ("ATTACK", "AGENT"):
                visible.pop()
            visible.append(
                {
                    "tag": "IS_ATTACK_REFUSED",
                    "gen_id": row["gen_id"],
                    "text": "❌ ATTACK REFUSED → ROLLBACK",
                }
            )
        else:
            visible.append(row)

        if tag == "EVAL":
            metrics.append(json.loads(row["text"]))

    return visible, metrics


# -------------------------
# Render
# -------------------------
current_frame = 0


def render(frame):
    ax_timeline.clear()
    ax_text.clear()
    ax_metrics.clear()

    visible, metrics = simulate_until(frame)

    # timeline
    ax_timeline.set_title(f"Workflow — step {frame}")
    ax_timeline.set_yticks(range(len(models)))
    ax_timeline.set_yticklabels(models)

    for i, r in enumerate(visible):
        ax_timeline.scatter(
            i, list(models).index(r["gen_id"]), s=200, color=colors.get(r["tag"], "black")
        )

    # text
    ax_text.axis("off")
    last = visible[-1]
    ax_text.text(
        0.01, 0.95, f"{last['tag']} — {last['gen_id']}", fontsize=11, weight="bold", va="top"
    )
    ax_text.text(0.01, 0.88, last["text"], fontsize=9, va="top", wrap=True)

    # metrics
    ax_metrics.set_title("Evaluation metrics")
    ax_metrics.set_ylim(0, 5)
    if metrics:
        dfm = pd.DataFrame(metrics)
        for k in ["proximity", "erosion", "cooperation"]:
            ax_metrics.plot(dfm[k], label=k)
        ax_metrics.legend()

    fig.canvas.draw_idle()


# -------------------------
# Buttons (SAFE positions)
# -------------------------
ax_prev = fig.add_axes([0.70, 0.02, 0.10, 0.06])
ax_next = fig.add_axes([0.82, 0.02, 0.10, 0.06])

btn_prev = Button(ax_prev, "◀ Prev")
btn_next = Button(ax_next, "Next ▶")

btn_prev.on_clicked(lambda _: render_step(-1))
btn_next.on_clicked(lambda _: render_step(+1))


def render_step(delta):
    global current_frame
    current_frame = max(0, min(len(df) - 1, current_frame + delta))
    render(current_frame)


# -------------------------
# Keyboard fallback (ALWAYS works)
# -------------------------
def on_key(event):
    if event.key == "right":
        render_step(+1)
    elif event.key == "left":
        render_step(-1)


fig.canvas.mpl_connect("key_press_event", on_key)

# -------------------------
# Start
# -------------------------
render(0)
plt.show()
