from core.attacks.datastore.read import get_data, Frame

import os
from copy import deepcopy
import json
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="make_and_split_dataset",
        description="Takes an Ares dataset path and turns it into a jsonl ingestible by the trainer",
        epilog="Use this for fine-tuning against Ares attacks",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset folder",
        default="./datasets/crescendo_bss_gemini_vs_circuitbreaker_ebunpbnfra",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output folder",
        default="train/data/all",
    )
    parser.add_argument(
        "-t", "--tag", type=str, help="Attack tag (lowercase), for example 'crescendo'"
    )
    parser.add_argument("--suffix", type=str, default="1", help="Suffix: save = tag + suffix")
    parser.add_argument("-s", "--seed", type=int, help="Rng seed for split generation", default=42)
    parser.add_argument("--trainsplit", type=int, default=80, help="train-split percent")
    parser.add_argument(
        "--replace-refusals",
        type=str,
        default="remove",
        help='If "remove" it will remove refused frames, alse it will replace the answer with the provided sorry message',
    )
    args = parser.parse_args()
    directory_path = args.dataset or "./datasets/crescendo_bss_gemini_vs_circuitbreaker_ebunpbnfra"
    entries = os.listdir(directory_path)
    random.seed(args.seed)
    train_entries = random.sample(entries, k=(len(entries) * args.trainsplit) // 100)
    all_dirs = []
    all_test_dirs = []
    for entry_name in train_entries:
        full_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(full_path):
            all_dirs.append(entry_name)
    for entry_name in entries:
        if entry_name in all_dirs:
            continue
        full_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(full_path):
            all_test_dirs.append(entry_name)
    all_data: list[Frame] = []
    for subdir in all_dirs:
        data: list[Frame] = get_data(subdir, directory_path, str(args.tag).lower())
        clean_data = [d for d in data if (d.has_attack_query() and not d.is_done())]
        all_data += clean_data

    all_test_data: list[Frame] = []
    for subdir in all_test_dirs:
        data: list[Frame] = get_data(subdir, directory_path, str(args.tag).lower())
        clean_data = [d for d in data if (d.has_attack_query() and not d.is_done())]
        all_test_data += clean_data

    print(f"{len(all_data)=} {len(all_test_data)=}")

    with open(f"{args.output}/{args.tag}:{args.suffix}_train.jsonl", "w") as f:
        last_runid = None
        last_conversation = None
        turn_acc = []
        all_objs = []
        for d in all_data:
            if last_runid != d.run_id:
                turn_acc = []
                last_conversation = None
            try:
                attack_q = d.attack_query()
                if d.is_empty_frame():
                    print("Empty frame skipped!")
                    last_runid = d.run_id
                    continue
                agent_r = d.agent_response()
                eval_ = d.response_evaluation_or_refusal()

                eval_ = {
                    "proximity": eval_.get("proximity", 0),
                    "erosion": eval_.get("erosion", 0),
                    "cooperation": eval_.get("cooperation", 0),
                    "harmful": eval_.get("goal_revealed", False),
                }

                convo = d.agent_conversation()

                if d.has_agent_response():
                    if convo is not None and last_conversation is not None:
                        rollback_and_new = convo.size() - last_conversation.size()
                        if not (rollback_and_new in [0, 2]):
                            print(
                                f"Possible bad rollback detected: {rollback_and_new} {last_conversation.size()}->{convo.size()} {d.run_id}"
                            )
                        rollback = rollback_and_new - 2

                        while rollback < 0:
                            turn_acc.pop()
                            turn_acc.pop()
                            rollback += 2

                object_user = {"role": "user", "content": attack_q}
                object_assistant = {"role": "assistant", "content": agent_r, "eval": eval_}
                turn_acc.append(object_user)
                turn_acc.append(object_assistant)
                json.dump(
                    {
                        "conversation_id": d.run_id,
                        "conversation_length": len(turn_acc) // 2,
                        "turns": deepcopy(turn_acc),
                    },
                    f,
                )
                f.write("\n")
                last_conversation = convo
            except Exception as e:
                print(f"skipped because: {e}")
                last_conversation = None
            last_runid = d.run_id

    with open(f"{args.output}/{args.tag}:{args.suffix}_test.jsonl", "w") as f:
        last_runid = None
        turn_acc = []
        all_objs = []
        for d in all_test_data:
            if last_runid != d.run_id:
                turn_acc = []
                last_conversation = None
            try:
                attack_q = d.attack_query()
                if d.is_empty_frame():
                    print("Empty frame skipped!")
                    last_runid = d.run_id
                    continue
                agent_r = d.agent_response()
                eval_ = d.response_evaluation_or_refusal()

                eval_ = {
                    "proximity": eval_.get("proximity", 0),
                    "erosion": eval_.get("erosion", 0),
                    "cooperation": eval_.get("cooperation", 0),
                    "harmful": eval_.get("goal_revealed", False),
                }

                convo = d.agent_conversation()

                if d.has_agent_response():
                    if convo is not None and last_conversation is not None:
                        rollback_and_new = convo.size() - last_conversation.size()
                        if not (rollback_and_new in [0, 2]):
                            print(
                                f"Possible bad rollback detected: {rollback_and_new} {last_conversation.size()}->{convo.size()} {d.run_id}"
                            )
                        rollback = rollback_and_new - 2

                        while rollback < 0:
                            turn_acc.pop()
                            turn_acc.pop()
                            rollback += 2

                object_user = {"role": "user", "content": attack_q}
                object_assistant = {"role": "assistant", "content": agent_r, "eval": eval_}
                turn_acc.append(object_user)
                turn_acc.append(object_assistant)
                json.dump(
                    {
                        "conversation_id": d.run_id,
                        "conversation_length": len(turn_acc) // 2,
                        "turns": deepcopy(turn_acc),
                    },
                    f,
                )
                f.write("\n")
                last_conversation = convo
            except Exception as e:
                print(f"skipped because: {e}")
                last_conversation = None
            last_runid = d.run_id
