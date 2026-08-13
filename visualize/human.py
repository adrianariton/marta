from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import re
import json
from pathlib import Path
import sys
import warnings

app = Flask(__name__)
import argparse
import textwrap
from utils import Aider

parser = argparse.ArgumentParser(
    description="Process safety datasets and attack tags.",
    epilog=textwrap.dedent(
        """\
        Example usage:
          python human.py --base_folder ./data --tag GOAT
          python human.py -dt CRESCENDO
          python human.py -dt GOAT -s adv
          python human.py --help
        """
    ),
)

# --base_folder / -b
parser.add_argument(
    "--base_folder",
    "-b",
    type=str,
    default="../datasets/crescendo_bss_gemini_vs_circuitbreaker",
    help="Path to the base folder containing datasets (default: ../datasets/crescendo_bss_gemini_vs_circuitbreaker)",
)

# --tag / -t
parser.add_argument(
    "--tag", "-t", type=str, default="ATTACK", help="The attack tag to look for (default: ATTACK)"
)

parser.add_argument(
    "--deduce_base_folder",
    "-d",
    action="store_true",
    help="Enable automatic deduction of the base folder path. Attack will be deduced as ../datasets/<tag.lower()>_<dataset>_gemini_vs_circuitbreaker",
)

parser.add_argument("--dataset", "-s", type=str, default="bss", help="The dataset: adv/bss")

parser.add_argument("--port", "-p", type=int, default=5000, help="The port to run the server.")

args = parser.parse_args()

# Configure your base folder here
tag_ATTACK = args.tag  # "ATTACK" if len(sys.argv) < 3 else sys.argv[2]

BASE_FOLDER = (
    args.base_folder
    if not args.deduce_base_folder
    else f"../datasets/{tag_ATTACK.lower()}_bss_gemini_vs_circuitbreaker"
)  # Change this to your actual path

aider = Aider(model="gemini-2.5-flash")
import ast


def get_folders():
    folders = []
    if os.path.exists(BASE_FOLDER):
        folders = [
            f for f in os.listdir(BASE_FOLDER) if os.path.isdir(os.path.join(BASE_FOLDER, f))
        ]
        folders = sorted(folders)
        is_jailbroken = []
        for f in folders:
            file_path = os.path.join(BASE_FOLDER, f, "info.txt")

            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = file.readline()
                    data = data.replace("'", '"')
                    data = ast.literal_eval(data)
                    # Assuming '1' or 'True' in the file means jailbroken
                    is_jailbroken.append(len(list(data.keys())) > 0)
            else:
                is_jailbroken.append(False)
        folders = [(f, z) for f, z in zip(folders, is_jailbroken)]
    return folders


def get_data(folder_name):
    """Get all JSONL data from a folder, organized by run_id and batched by tag sequences"""
    folder_path = os.path.join(BASE_FOLDER, folder_name)

    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}

    def get_shard(f):
        m = re.search(r"shard=(\d+)", f)
        return int(m.group(1)) if m else None

    # Get all JSONL files
    jsonl_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".jsonl")], key=lambda f: get_shard(f)
    )

    # Organize data by run_id
    runs = {}

    for file in jsonl_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                run_id = entry.get("run_id")

                if run_id not in runs:
                    runs[run_id] = []

                runs[run_id].append(entry)

    # Convert to frames format - each run_id becomes multiple frames (batches)
    frames = []

    for run_id, entries in runs.items():
        # Group entries into batches based on ATTACK tags (each ATTACK starts a new batch)
        batches = []
        current_batch = []

        for entry in entries:
            tag = entry.get("tag", "")

            # ATTACK tag starts a new batch
            if tag.lower() == tag_ATTACK.lower():
                if current_batch:  # Save previous batch if it exists
                    batches.append(current_batch)
                current_batch = [entry]
            else:
                current_batch.append(entry)

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        # Convert each batch to a frame
        for batch_idx, batch in enumerate(batches):
            frame = {
                "run_id": run_id,
                "batch_idx": batch_idx,
                "attack": None,
                "agent": None,
                "is_attack_refused": None,
                "eval": None,
            }

            for entry in batch:
                tag = entry.get("tag", "").lower()
                text = entry.get("text", "")

                # Handle the text field (which contains tuple: conversation list and reply)
                if isinstance(text, list) and len(text) == 2:
                    conversation, reply = text
                    goal = "unspecified"
                    if isinstance(conversation, dict):
                        if "goal" in conversation:
                            goal = conversation["goal"]
                        if "conversation" in conversation:
                            conversation = conversation["conversation"]
                    else:
                        warnings.warn(
                            "Old or invalid format detected. Proceeding anyway. Consider using new format."
                        )
                    frame[tag] = {
                        "conversation": conversation,
                        "reply": reply,
                        "raw": entry,
                        "goal": goal,
                    }
                else:
                    frame[tag] = {
                        "conversation": [],
                        "reply": str(text),
                        "raw": entry,
                        "goal": goal,
                    }
            if not (tag_ATTACK.lower() in frame.keys()):
                continue
            # if frame[tag_ATTACK.lower()]["reply"] == ["(nothing)"]:
            #     continue
            frames.append(frame)
        if len(frames) > 0:
            frames[-1]["is_final"] = True

    return {"frames": frames, "total": len(frames), "tag_attack": tag_ATTACK.lower()}


import random

random.seed(42)
tag_AGENT = "agent"


def convo_strip(c):
    if isinstance(c, list) and isinstance(c[0], dict) and "conversation" in c[0]:
        return c[0]
    return c


def get_all_frames(convo_max: int = None, goal_revealed: bool = None):
    folders = get_folders()
    folder_names = [x[0] for x in folders]
    frames = []
    for f in folder_names:
        if len(frames) > 0:
            frames[-1]["is_final"] = True
        frame_dict = get_data(f)
        # print()
        # print()
        for fr in frame_dict["frames"]:

            agent_data = fr.get(tag_AGENT) or {}
            if convo_max is None or (
                "conversation" in agent_data
                and len(convo_strip(fr[tag_AGENT]["conversation"])) > convo_max
            ):
                frames += [fr]
    print(f"Total frames after filtering by convo_max={convo_max}: {len(frames)}")
    return frames


@app.route("/")
def index():
    return render_template("human.html", folder_name="Human Expert Labeling")


@app.route("/plot")
def plot_():
    return render_template("plot.html", folder_name="Real Plots")


@app.post("/api/eval/help/<model>")
def help_me_rate(model="gemini-2.5-flash"):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON provided"}), 400
    user_query = data.get("query")
    system = data.get("system")

    print(f"{user_query=} {system=}")
    try:
        return jsonify({"feedback": aider.aid_me_please(system, user_query, model=model.strip())})
    except Exception as e:
        return jsonify({"error": f"{e}"}), 400


@app.get("/api/get_some/<cmax_and_sample_size>")
def get_some_frames(cmax_and_sample_size="8:100"):

    random.seed(42)

    cmax, k = cmax_and_sample_size.split(":")
    frames = get_all_frames(convo_max=int(cmax))
    get_final = False

    def is_goal_rev(f):

        ff = f["eval"]
        if ff is None:
            return False

        eval_ = ff.get("reply", {})
        goal_rev = eval_.get("goal_revealed", False)
        return goal_rev

    def is_ref(f):
        ff = f["is_attack_refused"]
        if ff is None:
            return False
        if "is_attack_refused" not in f:
            return False
        if "reply" not in ff:
            return False
        return f["is_attack_refused"].get("reply", False) == True

    if get_final:
        print(f"{set([f.get('is_final') for f in frames])=}")
        frames = [f for f in frames if f.get("is_final", False)]
        print(f"Frames after filtering for final: {len(frames)}")

    # print(f"{set([get_ref(f) for f in frames])=}")

    final_frames = [f for f in frames if is_goal_rev(f)]
    non_final_frames = [f for f in frames if not is_ref(f) and not is_goal_rev(f)]
    print(f"Frames: {len(frames)}")
    print(f"Frames with goal revealed: {len(final_frames)}")
    print(f"Frames without goal revealed: {len(non_final_frames)}")

    final_frames = random.sample(final_frames, k=min(int(k) // 2, len(final_frames)))
    non_final_frames = random.sample(non_final_frames, k=min(int(k) // 2, len(non_final_frames)))
    all_frames = final_frames + non_final_frames
    random.shuffle(all_frames)
    return jsonify(
        {
            "frames": all_frames,
            "total": len(frames),
            "tag_attack": tag_ATTACK.lower(),
        }
    )


@app.get("/api/get_all")
def get_frames():
    frames = get_all_frames()
    evals = []
    for fr in frames:
        eval_ = fr.get("eval") or {}
        if eval_.get("reply") is not None:
            ev = eval_["reply"]
            evals.append(ev)
        else:
            evals.append({})
    return jsonify(
        {"frames": frames, "evals": evals, "total": len(frames), "tag_attack": tag_ATTACK.lower()}
    )


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
