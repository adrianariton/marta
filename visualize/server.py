from flask import Flask, render_template, jsonify, send_from_directory
import os
import re
import json
from pathlib import Path
import sys
import warnings

app = Flask(__name__)
import argparse
import textwrap

parser = argparse.ArgumentParser(
    description="Process safety datasets and attack tags.",
    epilog=textwrap.dedent(
        """\
        Example usage:
          python server.py --base_folder ./data --tag GOAT
          python server.py -dt CRESCENDO
          python server.py -dt GOAT -s adv
          python server.py -dt GOAT -s adv -v 1
          python server.py --help
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
# --tag / -t
parser.add_argument(
    "--version",
    "-v",
    type=str,
    default="",
    help="Version of the file: get's concatted to the file name __<v>",
)

parser.add_argument(
    "--target",
    "-g",
    type=str,
    default="circuitbreaker",
    help="Target: circuitbreaker|derta",
)

parser.add_argument(
    "--deduce_base_folder",
    "-d",
    action="store_true",
    help="Enable automatic deduction of the base folder path. Attack will be deduced as ../datasets/<tag.lower()>_<dataset>_<attacker>_vs_<target>__<v> or ../datasets/<tag.lower()>_<dataset>_<attacker>_vs_<target> if v is unset",
)

parser.add_argument("--dataset", "-s", type=str, default="bss", help="The dataset: adv/bss")

parser.add_argument(
    "--attacker", "-a", type=str, default="gemini", help="The attacker: gemini/mistral"
)

parser.add_argument("--port", "-p", type=int, default=5000, help="The port to run the server.")

args = parser.parse_args()

tag_ATTACK = args.tag
_base_folder = (
    args.base_folder
    if not args.deduce_base_folder
    else f"../datasets/{tag_ATTACK.lower()}_{args.dataset}_{args.attacker}_vs_{args.target}"
)
BASE_FOLDER = (
    _base_folder if args.version == "" else (_base_folder + "__" + args.version)
)  # Change this to your actual path
print(f"Using BASE_FOLDER: {BASE_FOLDER}")
import ast


@app.route("/")
def index():
    """Show all folders in BASE_FOLDER"""
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
                is_jailbroken.append(None)
        folders = [(f, z) for f, z in zip(folders, is_jailbroken)]
    return render_template(
        "index.html",
        folders=folders,
        bfolder=BASE_FOLDER,
    )


@app.route("/folder/<folder_name>")
def view_folder(folder_name):
    """Show viewer for a specific folder"""
    return render_template("viewer.html", folder_name=folder_name)


@app.route("/api/data/<folder_name>")
def get_data(folder_name):
    """Get all JSONL data from a folder, organized by run_id and batched by tag sequences"""
    folder_path = os.path.join(BASE_FOLDER, folder_name)

    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder not found"}), 404

    def get_shard(f):
        m = re.search(r"shard=(\d+)", f)
        return int(m.group(1)) if m else None

    # Get all JSONL files
    jsonl_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".jsonl")], key=lambda f: get_shard(f)
    )

    print(jsonl_files)

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

        print(f"{len(batches)=} {len(entries)=}")

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

    return jsonify({"frames": frames, "total": len(frames), "tag_attack": tag_ATTACK.lower()})

import re
 
# ── ADD this block after your existing imports / before @app.route("/") ────
import os
import re
import ast
import json
import warnings
 
# Lazy-load datasets once
_datasets_loaded = False
adv_df = lao_df = sst_df = xss_df = hb_df = sorry_df = None
 
def _load_datasets():
    global _datasets_loaded, adv_df, lao_df, sst_df, xss_df, hb_df, sorry_df
    if _datasets_loaded:
        return
    from core.datasets.safety.harmbench import HarmBench
    from core.datasets.safety.advbench import AdvBench
    from core.datasets.safety.donotanswer import LAODoNotAnswer
    from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
    from core.datasets.safety.xstest import XSTest
    from core.datasets.safety.sorrybench import SorryBench
    adv_df    = AdvBench().get_first(n=1000)
    lao_df    = LAODoNotAnswer().get_first(n=1000)
    sst_df    = BertievidgenSimpleSafetyTests().get_first(n=1000)
    xss_df    = XSTest().get_first(1000)
    hb_df     = HarmBench().get_first(1000)
    sorry_df  = SorryBench().get_first(500)
    _datasets_loaded = True
 
 
def _get_goal(ds, _id):
    """Given a dataset name and raw id string, return the goal/prompt text."""
    _load_datasets()
    try:
        if ds == "walledai/AdvBench":
            df = adv_df
            match = re.search(r"adv_\d{3}", _id)
            _id = match.group()
            q_col = "prompt"
        elif ds == "Bertievidgen/SimpleSafetyTests":
            df = sst_df
            match = re.search(r"sst_\d{3}", _id)
            _id = match.group()
            q_col = "prompt"
        elif ds == "LibrAI/do-not-answer":
            df = lao_df
            match = re.search(r"lao_(\d+)", _id)
            _id = int(match.group(1))
            q_col = "question"
        elif ds == "walledai/XSTest":
            df = xss_df
            match = re.search(r"xstest_(?:safe|unsafe)_\d+", _id)
            _id = match.group()
            q_col = "prompt"
        elif ds == "walledai/HarmBench":
            df = hb_df
            match = re.search(r"hb_(\d+)", _id)
            _id = "hb_" + match.group(1)
            q_col = "Behavior"
        elif ds == "sorry-bench/sorry-bench-202503":
            df = sorry_df
            match = re.search(r"(\d+)", _id)
            _id = "sorry_" + match.group(1)
            q_col = "prompt"
        else:
            return None
        return df[df["id"] == _id][q_col].iloc[0]
    except Exception as e:
        print(f"[get_goal] {e} | ds={ds} id={_id}")
        return None
 
 
def _goal_from_folder_name(folder_name: str):
    """
    Folder names look like:
      sorry_42__run3  /  hb_007__run1  /  adv_001  etc.
    We infer the dataset from the id prefix.
    """
    DS_MAP = {
        r"adv_":            "walledai/AdvBench",
        r"sst_":            "Bertievidgen/SimpleSafetyTests",
        r"lao_":            "LibrAI/do-not-answer",
        r"xstest_":         "walledai/XSTest",
        r"hb_":             "walledai/HarmBench",
        r"sorry_":          "sorry-bench/sorry-bench-202503",
    }
    for prefix, ds in DS_MAP.items():
        if re.search(prefix, folder_name):
            return _get_goal(ds, folder_name)
    return None
 
 
# ── ADD this new route anywhere after the existing routes ───────────────────
@app.route("/api/goal/<path:folder_name>")
def get_goal_api(folder_name):
    """Return the goal/prompt for a given folder name."""
    goal = _goal_from_folder_name(folder_name)
    if goal is None:
        return jsonify({"goal": "Goal not found"}), 200
    return jsonify({"goal": str(goal)})

if __name__ == "__main__":
    app.run(debug=True, port=args.port)
