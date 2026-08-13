from abc import ABC, abstractmethod


class Attack(ABC):
    @abstractmethod
    def get_base_folder(self, index):
        pass

    @abstractmethod
    def get_subfolder(self, index):
        pass

    @abstractmethod
    def attack_id(self):
        pass


def get_against_name(index):
    if index == 0:
        return "circuitbreaker"
    elif index == 1:
        return "dertains"
    elif index == 2:
        return "door"
    elif index == 3:
        return "guardllama"
    elif index == 4:
        return "llamainstruct"
    elif index == 5:
        return "cbllama"
    else:
        raise ValueError("Invalid index")


class A_Crescendo(Attack):
    def get_base_folder(self, index):
        if index == 0:
            return "datasets/crescendo_xstest_qwen25_vs_circuitbreaker__async_batch_bagel_eval_3repl_3x8"
        elif index == 1:
            return "datasets/crescendo_xstest_qwen25_vs_dertains__async_batch_bagel_eval_3repl_3x8"
        elif index == 2:
            return "datasets/crescendo_xstest_qwen25_vs_door__async_batch_bagel_eval_3repl_3x8"
        elif index == 3:
            return (
                "datasets/crescendo_xstest_qwen25_vs_guardllama__async_batch_bagel_eval_3repl_3x8"
            )
        elif index == 4:
            return "datasets/crescendo_xstest_qwen25_vs_llamainstruct__async_batch_bagel_eval_3repl_3x8"
        elif index == 5:
            return "datasets/crescendo_xstest_qwen25_vs_cbllama__async_batch_bagel_eval_3repl_3x8"

        else:
            raise ValueError("Invalid index")

    def get_subfolder(self, index):
        return "crescendo_bss_" + index

    def attack_id(self):
        return "crescendo"


class A_FITD(Attack):
    def get_base_folder(self, index):
        if index == 0:
            return "datasets/fitd_xstest_qwen25_vs_circuitbreaker__async_batch_bagel_eval_3repl"
        elif index == 1:
            return "datasets/fitd_xstest_qwen25_vs_dertains__async_batch_bagel_eval_3repl"
        elif index == 2:
            return "datasets/fitd_xstest_qwen25_vs_door__async_batch_bagel_eval_3repl"
        elif index == 3:
            return "datasets/fitd_xstest_qwen25_vs_guardllama__async_batch_bagel_eval_3repl"
        elif index == 4:
            return "datasets/fitd_xstest_qwen25_vs_llamainstruct__async_batch_bagel_eval_3repl"
        elif index == 5:
            return "datasets/fitd_xstest_qwen25_vs_cbllama__async_batch_bagel_eval_3repl"
        else:
            raise ValueError("Invalid index")

    def get_subfolder(self, index):
        return "fitd_bss_" + index

    def attack_id(self):
        return "fitd"


class A_Goat(Attack):
    def get_base_folder(self, index):
        if index == 0:
            return "datasets/goat_xstest_qwen25_vs_circuitbreaker__async_batch_bagel_eval_3repl"
        elif index == 1:
            return "datasets/goat_xstest_qwen25_vs_dertains__async_batch_bagel_eval_3repl"
        elif index == 2:
            return "datasets/goat_xstest_qwen25_vs_door__async_batch_bagel_eval_3repl"
        elif index == 3:
            return "datasets/goat_xstest_qwen25_vs_guard__async_batch_bagel_eval_3repl"
        elif index == 4:
            return "datasets/goat_xstest_qwen25_vs_llamainstruct__async_batch_bagel_eval_3repl"
        elif index == 5:
            return "datasets/goat_xstest_qwen25_vs_cbllama__async_batch_bagel_eval_3repl"
        else:
            raise ValueError("Invalid index")

    def get_subfolder(self, index):
        return "goat_bss_" + index

    def attack_id(self):
        return "goat"


class A_RACE(Attack):
    def __init__(self, suffix=""):
        super().__init__()
        self.suffix = suffix

    def get_base_folder(self, index):
        if index == 0:
            return (
                "datasets/race_xstest_qwen25_vs_circuitbreaker__async_batch_bagel_eval_3repl"
                + self.suffix
            )
        elif index == 1:
            return (
                "datasets/race_xstest_qwen25_vs_dertains__async_batch_bagel_eval_3repl"
                + self.suffix
            )
        elif index == 2:
            return "datasets/race_xstest_qwen25_vs_door__async_batch_bagel_eval_3repl" + self.suffix
        elif index == 3:
            return (
                "datasets/race_xstest_qwen25_vs_guardllama__async_batch_bagel_eval_3repl"
                + self.suffix
            )
        elif index == 4:
            return (
                "datasets/race_xstest_qwen25_vs_llamainstruct__async_batch_bagel_eval_3repl"
                + self.suffix
            )
        elif index == 5:
            return (
                "datasets/race_xstest_qwen25_vs_cbllama__async_batch_bagel_eval_3repl" + self.suffix
            )
        else:
            raise ValueError("Invalid index")

    def get_subfolder(self, index):
        return "race_bss_" + index

    def attack_id(self):
        return "race"


xstest_group_keys = ["focus", "type"]

attacks: list[Attack] = [A_Crescendo(), A_FITD(), A_Goat(), A_RACE(suffix="_powerful")]

from core.datasets.safety.xstest import XSTest
import matplotlib.pyplot as plt
import os
import ast

df = XSTest().get_first(n=1000)


def plot_base_folder(base_folder, target_name, attack: Attack):
    datas = []
    for i, row in df.iterrows():
        if not ("unsafe" in row["id"]):
            continue

        # if not ("How" in row["prompt"]):
        #     continue

        output_folder_name = f"{base_folder}/" + attack.get_subfolder(row["id"])
        info = output_folder_name + "/info.txt"

        if os.path.exists(info):
            with open(info, "rt") as f:
                data = f.readline()
                data = data.replace("'", '"')
                data = ast.literal_eval(data)
                if len(list(data.keys())) == 0:
                    datas.append({"jailbroken": False, "data": data, **row.to_dict()})
                else:
                    first_key = sorted(list(data.keys()))[0]
                    dt = {first_key: data[first_key]}
                    datas.append({"jailbroken": True, "data": dt, **row.to_dict()})

        else:
            datas.append({"jailbroken": None, "data": {}, **row.to_dict()})
    assert len(datas) == 200, f"Expected 200 unsafe cases, got {len(datas)}"
    return datas


import pandas as pd


def plot_grouped(attack_index):
    for index in range(6):
        target_name = get_against_name(index)
        base_folder = attacks[attack_index].get_base_folder(index)
        attack_name = attacks[attack_index].__class__.__name__[2:].lower()
        print(f"Base Folder: {base_folder}")
        datas = plot_base_folder(base_folder, target_name, attacks[attack_index])
        ddf = pd.DataFrame(datas)
        pass_rate = ddf.groupby("type")["jailbroken"].mean() * 100
        pass_rate.sort_values(ascending=False).plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title(
            f"{attack_name.capitalize()} Jailbreak Rate (%) by Focus against {target_name.capitalize()}"
        )
        plt.ylabel("Success Rate (%)")
        plt.xlabel("Focus")
        plt.show()


def plot_grouped_in_one(attack_index):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for index in range(6):
        ax = axes[index]
        target_name = get_against_name(index)
        base_folder = attacks[attack_index].get_base_folder(index)
        attack_name = attacks[attack_index].__class__.__name__[2:].lower()
        print(f"Base Folder: {base_folder}")
        datas = plot_base_folder(base_folder, target_name, attacks[attack_index])
        ddf = pd.DataFrame(datas)
        pass_rate = ddf.groupby("type")["jailbroken"].mean() * 100
        pass_rate.sort_values(ascending=False).plot(
            kind="bar", color="skyblue", edgecolor="black", ax=ax
        )
        ax.set_title(
            f"{attack_name.capitalize()} Jailbreak Rate (%) by Focus against {target_name.capitalize()}"
        )
        ax.set_ylabel("Success Rate (%)")
        ax.set_xlabel("Focus")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Hide the 6th subplot if only 5 plots (change range to 5 if needed)
    # axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"graphs/{attack_name}_all_targets.png")
    plt.show()


def plot_grouped_in_one_wm(attack_index):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for index in range(6):
        ax = axes[index]
        target_name = get_against_name(index)
        base_folder = attacks[attack_index].get_base_folder(index)
        attack_name = attacks[attack_index].__class__.__name__[2:].lower()
        print(f"Base Folder: {base_folder}")
        datas = plot_base_folder(base_folder, target_name, attacks[attack_index])
        ddf = pd.DataFrame(datas)

        total = ddf.groupby("type")["jailbroken"].count()
        jailbroken = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).sum())
        missing = ddf.groupby("type")["jailbroken"].apply(lambda x: x.isna().sum())

        jailbroken_rate = jailbroken / total * 100
        missing_rate = missing / total * 100

        # Sort by jailbroken rate descending
        order = jailbroken_rate.sort_values(ascending=False).index
        jailbroken_rate = jailbroken_rate[order]
        missing_rate = missing_rate[order]

        x = range(len(order))
        ax.bar(x, jailbroken_rate, color="skyblue", edgecolor="black", label="Jailbroken")
        ax.bar(
            x,
            missing_rate,
            bottom=jailbroken_rate,
            color="red",
            edgecolor="black",
            alpha=0.7,
            label="Missing (None)",
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_title(
            f"{attack_name.capitalize()} Jailbreak Rate (%) by Focus against {target_name.capitalize()}"
        )
        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Focus")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"graphs/{attack_name}_all_targets.png")
    plt.show()


def plot_grouped_in_one_wmn(attack_index):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for index in range(6):
        ax = axes[index]
        target_name = get_against_name(index)
        base_folder = attacks[attack_index].get_base_folder(index)
        attack_name = attacks[attack_index].__class__.__name__[2:].lower()
        print(f"Base Folder: {base_folder}")
        datas = plot_base_folder(base_folder, target_name, attacks[attack_index])
        ddf = pd.DataFrame(datas)

        total = ddf.groupby("type")["jailbroken"].size()
        jailbroken = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).sum())
        not_jailbroken = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == False).sum())
        missing = ddf.groupby("type")["jailbroken"].apply(lambda x: x.isna().sum())

        jailbroken_rate = jailbroken / total * 100
        not_jailbroken_rate = not_jailbroken / total * 100
        missing_rate = missing / total * 100

        order = jailbroken_rate.sort_values(ascending=False).index
        jailbroken_rate = jailbroken_rate[order]
        not_jailbroken_rate = not_jailbroken_rate[order]
        missing_rate = missing_rate[order]

        x = range(len(order))
        ax.bar(x, jailbroken_rate, color="skyblue", edgecolor="black", label="Jailbroken")
        ax.bar(
            x,
            not_jailbroken_rate,
            bottom=jailbroken_rate,
            color="gray",
            edgecolor="black",
            alpha=0.7,
            label="Not Jailbroken",
        )
        ax.bar(
            x,
            missing_rate,
            bottom=jailbroken_rate + not_jailbroken_rate,
            color="red",
            edgecolor="black",
            alpha=0.7,
            label="Missing (None)",
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_title(
            f"{attack_name.capitalize()} Jailbreak Rate (%) by Focus against {target_name.capitalize()}"
        )
        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Focus")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"graphs/{attack_name}_all_targets.png")
    plt.show()


def plot_grouped_in_one_wmnc(attack_index):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for index in range(6):
        ax = axes[index]
        target_name = get_against_name(index)
        base_folder = attacks[attack_index].get_base_folder(index)
        attack_name = attacks[attack_index].__class__.__name__[2:].lower()
        print(f"Base Folder: {base_folder}")
        datas = plot_base_folder(base_folder, target_name, attacks[attack_index])
        ddf = pd.DataFrame(datas)

        total = ddf.groupby("type")["jailbroken"].size()
        jailbroken = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).sum())
        not_jailbroken = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == False).sum())
        missing = ddf.groupby("type")["jailbroken"].apply(lambda x: x.isna().sum())

        jailbroken_rate = jailbroken / total * 100
        not_jailbroken_rate = not_jailbroken / total * 100
        missing_rate = missing / total * 100

        order = jailbroken_rate.sort_values(ascending=False).index
        jailbroken_rate = jailbroken_rate[order]
        not_jailbroken_rate = not_jailbroken_rate[order]
        missing_rate = missing_rate[order]
        total = total[order]

        x = range(len(order))
        ax.bar(x, jailbroken_rate, color="skyblue", edgecolor="black", label="Jailbroken")
        ax.bar(
            x,
            not_jailbroken_rate,
            bottom=jailbroken_rate,
            color="gray",
            edgecolor="black",
            alpha=0.7,
            label="Not Jailbroken",
        )
        ax.bar(
            x,
            missing_rate,
            bottom=jailbroken_rate + not_jailbroken_rate,
            color="red",
            edgecolor="black",
            alpha=0.7,
            label="Missing (None)",
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{t}\n(n={total[t]})" for t in order], rotation=45, ha="right")
        ax.set_title(
            f"{attack_name.capitalize()} Jailbreak Rate (%) by Focus against {target_name.capitalize()}"
        )
        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Focus")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"graphs/{attack_name}_all_targets.png")
    plt.show()


for i in range(4):
    plot_grouped_in_one_wmnc(i)


def print_latex_tables():
    for attack_index, attack_obj in enumerate(attacks):
        attack_name = attack_obj.__class__.__name__[2:].capitalize()

        for index in range(6):
            target_name = get_against_name(index).capitalize()
            base_folder = attack_obj.get_base_folder(index)

            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            stats = ddf.groupby("type")["jailbroken"].agg(
                Total="size", Jailbroken=lambda x: (x == True).sum()
            )
            stats["JB_Rate"] = (stats["Jailbroken"] / stats["Total"] * 100).round(1)
            stats = stats.sort_values("JB_Rate", ascending=False)

            print(f"\n% --- {attack_name} vs {target_name} ---")
            print(r"\begin{table}[t]")
            print(f"\\caption{{Jailbreak Rate by Focus: {attack_name} vs {target_name}}}")
            print(f"\\label{{tab:{attack_name.lower()}_{target_name.lower()}}}")
            print(r"\begin{center}")
            print(r"\begin{tabular}{ll}")
            print(r"\multicolumn{1}{c}{\bf FOCUS TYPE}  &\multicolumn{1}{c}{\bf SUCCESS RATE (\%)}")
            print(r"\\ \hline \\")

            for focus, row in stats.iterrows():
                print(f"{focus.replace('_', ' '):<20} & {row['JB_Rate']}\\% \\\\")

            print(r"\end{tabular}")
            print(r"\end{center}")
            print(r"\end{table}")


def print_joint_asr_tables():
    print("\n% ================= JOINT ASR TABLES =================")
    for attack_index, attack_obj in enumerate(attacks):
        attack_name = attack_obj.__class__.__name__[2:].capitalize()
        target_results = []

        for index in range(6):
            target_name = get_against_name(index).capitalize()
            base_folder = attack_obj.get_base_folder(index)

            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            # Overall ASR calculation (Successful / Total attempted)
            total = len(ddf)
            assert total == 200, f"Expected 200 unsafe cases for overall ASR, got {total}"
            successful = (ddf["jailbroken"] == True).sum()
            asr = (successful / total * 100) if total > 0 else 0
            target_results.append((target_name, asr))

        print(f"\n% Joint Table for {attack_name}")
        print(r"\begin{table}[t]")
        print(f"\\caption{{Overall ASR for {attack_name} across all targets}}")
        print(f"\\label{{tab:joint_asr_{attack_name.lower()}}}")
        print(r"\begin{center}")
        print(r"\begin{tabular}{ll}")
        print(r"\multicolumn{1}{c}{\bf TARGET MODEL}  &\multicolumn{1}{c}{\bf OVERALL ASR (\%)}")
        print(r"\\ \hline \\")

        for target, asr in target_results:
            print(f"{target:<20} & {asr:.1f}\\% \\\\")

        print(r"\end{tabular}")
        print(r"\end{center}")
        print(r"\end{table}")


# Execute at the end of your script
# print_latex_tables()
# print_joint_asr_tables()

# Call this at the bottom of your script
# print_latex_tables()
print_latex_tables()
print_joint_asr_tables()


def print_type_breakdown_table():
    print("\n% ================= TYPE-BASED BREAKDOWN TABLE =================")
    for attack_index, attack_obj in enumerate(attacks):
        attack_name = attack_obj.__class__.__name__[2:].capitalize()

        # Dictionary to store ASR per type per target
        # Structure: { type_name: { target_name: asr_value } }
        type_data = {}
        all_targets = [get_against_name(i).capitalize() for i in range(6)]

        for index in range(6):
            target_name = all_targets[index]
            base_folder = attack_obj.get_base_folder(index)
            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            # Calculate ASR per type
            stats = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)

            for category, asr in stats.items():
                if category not in type_data:
                    type_data[category] = {}
                type_data[category][target_name] = asr

        # Sort types alphabetically for the table
        sorted_types = sorted(type_data.keys())

        print(f"\n% Type Breakdown for {attack_name}")
        print(r"\begin{table*}[t]")  # Using table* for wide content
        print(f"\\caption{{ASR (\%) by Prompt Category: {attack_name}}}")
        print(f"\\label{{tab:type_breakdown_{attack_name.lower()}}}")
        print(r"\begin{center}")
        # Define columns: Type + 6 targets
        print(r"\begin{tabular}{l" + "c" * 6 + "}")
        print(r"\hline")
        header = (
            r"\textbf{Category} & " + " & ".join([rf"\textbf{{{t}}}" for t in all_targets]) + r" \\"
        )
        print(header)
        print(r"\hline")

        for category in sorted_types:
            row_str = f"{category.replace('_', ' '):<20}"
            for target in all_targets:
                asr = type_data[category].get(target, 0.0)
                row_str += f" & {asr:.1f}\\%"
            print(row_str + r" \\")

        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{center}")
        print(r"\end{table*}")


# Add this to your execution block
print_type_breakdown_table()
print("********************************")


def print_comprehensive_joint_figure():
    print("\n% ================= COMPREHENSIVE JOINT FIGURE (4 SUBTABLES) =================")

    # We will store the results for all 4 attacks here to print in one block
    all_attack_matrices = []
    targets = [get_against_name(i).capitalize() for i in range(6)]

    for attack_index in range(4):
        attack_obj = attacks[attack_index]
        attack_name = attack_obj.__class__.__name__[2:].capitalize()
        matrix = {}

        for index in range(6):
            target_name = targets[index]
            base_folder = attack_obj.get_base_folder(index)
            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            # Group by type and calculate rates
            def calculate_rate(group):
                if group.isna().all():
                    return "--"
                return (group == True).mean() * 100

            cat_stats = ddf.groupby("type")["jailbroken"].apply(calculate_rate)

            for category, val in cat_stats.items():
                if category not in matrix:
                    matrix[category] = {}
                matrix[category][target_name] = val

        all_attack_matrices.append((attack_name, matrix))

    # --- START LATEX OUTPUT ---
    print(r"\begin{table*}[t]")
    print(r"\caption{Comprehensive Analysis of Jailbreak Success Rates (\%)}")
    print(r"\label{tab:combined_attacks}")
    print(r"\begin{center}")

    sorted_categories = sorted(all_attack_matrices[0][1].keys())

    for i, (name, matrix) in enumerate(all_attack_matrices):
        # Every 2 tables, we might want a bit of vertical space if they wrap
        if i == 2:
            print(r"\vspace{0.5cm}")

        print(f"% --- Subtable: {name} ---")
        print(r"\begin{subtable}{0.48\textwidth}")
        print(r"\centering")
        print(f"\\caption{{{name} Attack}}")
        # Narrower columns to fit 6 targets: CBM (Mistral), DER (Derta), DOR (Door), GUA (Guard), LLA (Llama), CBL (CbLlama)
        print(r"\begin{tabular}{lcccccc}")

        # Header
        short_targets = ["CBM", "DER", "DOR", "GUA", "LLA", "CBL"]
        header_parts = [r"\multicolumn{1}{c}{\bf CATEGORY}"]
        for st in short_targets:
            header_parts.append(rf"\multicolumn{{1}}{{c}}{{\bf {st}}}")
        print(" & ".join(header_parts) + r" \\")
        print(r"\hline \\")

        # Data Rows
        for cat in sorted_categories:
            display_name = cat.replace("_", " ").title()
            # Truncate long category names for space
            if len(display_name) > 15:
                display_name = display_name[:12] + "..."

            row_cells = [f"{display_name:<15}"]

            for target_idx, t_full_name in enumerate(targets):
                val = matrix[cat].get(t_full_name, "--")

                if val == "--":
                    cell_text = "--"
                else:
                    cell_text = f"{val:.1f}\\%"
                    if val > 50:
                        cell_text = f"\\textbf{{{cell_text}}}"

                row_cells.append(cell_text)
            print(" & ".join(row_cells) + r" \\")

        # Average Row
        print(r"\\ \hline \\")
        avg_row = [f"{'Average':<15}"]
        for t_full_name in targets:
            # Filter out the "--" strings to calculate numeric average
            numeric_vals = [
                matrix[cat].get(t_full_name)
                for cat in sorted_categories
                if isinstance(matrix[cat].get(t_full_name), (int, float))
            ]

            if not numeric_vals:
                avg_row.append("--")
            else:
                avg_total = sum(numeric_vals) / len(numeric_vals)
                avg_row.append(f"{avg_total:.1f}\\%")

        print(" & ".join(avg_row) + r" \\")
        print(r"\end{tabular}")
        print(r"\end{subtable}")

        # Add horizontal fill between left and right tables
        if i % 2 == 0:
            print(r"\hfill")
        else:
            print(r"\\ \vspace{0.3cm}")

    print(r"\end{center}")
    print(r"\end{table*}")


# Execute
print_comprehensive_joint_figure()
# def print_joint_categorical_asr_figure():
#     print("\n% ================= JOINT CATEGORICAL FIGURE (TABLE) =================")
#     for attack_index, attack_obj in enumerate(attacks):
#         attack_name = attack_obj.__class__.__name__[2:].capitalize()

#         # Structure: { type_name: { target_name: asr_value } }
#         matrix = {}
#         targets = [get_against_name(i).capitalize() for i in range(6)]

#         # Collect data for all targets for this specific attack
#         for index in range(6):
#             target_name = targets[index]
#             base_folder = attack_obj.get_base_folder(index)
#             datas = plot_base_folder(base_folder, target_name, attack_obj)
#             ddf = pd.DataFrame(datas)

#             # Calculate ASR per category
#             cat_asr = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)

#             for category, asr in cat_asr.items():
#                 if category not in matrix:
#                     matrix[category] = {}
#                 matrix[category][target_name] = asr

#         sorted_categories = sorted(matrix.keys())

#         print(f"\n% --- Joint Figure Table: {attack_name} ---")
#         print(r"\begin{table}[t]")
#         print(
#             f"\\caption{{Joint Analysis: Success Rate (\%) of {attack_name} across Focus Categories and Target Models.}}"
#         )
#         print(f"\\label{{tab:joint_{attack_name.lower()}}}")
#         print(r"\begin{center}")

#         # Defining columns: 1 left-aligned for category, 6 centered for targets
#         print(r"\begin{tabular}{lcccccc}")

#         # Header Row using \multicolumn and \bf to match your style
#         header_parts = [r"\multicolumn{1}{c}{\bf FOCUS CATEGORY}"]
#         for t in targets:
#             header_parts.append(rf"\multicolumn{{1}}{{c}}{{\bf {t.upper()}}}")

#         print(" & ".join(header_parts) + r" \\")
#         print(r"\hline \\")

#         # Data Rows
#         for cat in sorted_categories:
#             display_name = cat.replace("_", " ").title()
#             row_cells = [f"{display_name:<25}"]

#             for t in targets:
#                 val = matrix[cat].get(t, 0.0)
#                 cell_text = f"{val:.1f}\\%"

#                 # Apply textbf if value is > 50
#                 if val > 50:
#                     cell_text = f"\\textbf{{{cell_text}}}"

#                 row_cells.append(cell_text)

#             print(" & ".join(row_cells) + r" \\")

#         # Summary Average Row
#         print(r"\\ \hline \\")
#         avg_row = [f"{'Average ASR':<25}"]
#         for t in targets:
#             category_values = [matrix[cat].get(t, 0.0) for cat in sorted_categories]
#             avg_val = sum(category_values) / len(category_values) if category_values else 0
#             avg_row.append(f"{avg_val:.1f}\\%")

#         print(" & ".join(avg_row) + r" \\")

#         print(r"\end{tabular}")
#         print(r"\end{center}")
#         print(r"\end{table}")


# Call this alongside your other functions
# print_joint_categorical_asr_figure()


def print_vertical_comprehensive_figure():
    print(
        "\n% ================= VERTICAL COMPREHENSIVE FIGURE (4 TABLES STACKED) ================="
    )

    all_attack_matrices = []
    targets = [get_against_name(i).capitalize() for i in range(6)]

    for attack_index in range(4):
        attack_obj = attacks[attack_index]
        attack_name = attack_obj.__class__.__name__[2:].capitalize()
        matrix = {}

        for index in range(6):
            target_name = targets[index]
            base_folder = attack_obj.get_base_folder(index)
            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            def calculate_rate(group):
                if group.isna().all():
                    return "--"
                return (group == True).mean() * 100

            cat_stats = ddf.groupby("type")["jailbroken"].apply(calculate_rate)
            for category, val in cat_stats.items():
                if category not in matrix:
                    matrix[category] = {}
                matrix[category][target_name] = val

        all_attack_matrices.append((attack_name, matrix))

    print(r"\begin{table*}[p] % [p] to put on a dedicated results page")
    print(r"\caption{Comprehensive Analysis of Jailbreak Success Rates (\%)}")
    print(r"\begin{center}")

    sorted_categories = sorted(all_attack_matrices[0][1].keys())

    for i, (name, matrix) in enumerate(all_attack_matrices):
        print(f"% --- Subtable: {name} ---")
        # CHANGE 1: Set width to 1.0 (Full Page Width)
        print(r"\begin{subtable}{1.0\textwidth}")
        print(r"\centering")
        print(f"\\caption{{{name} Attack}}")
        print(r"\begin{tabular}{lcccccc}")

        # Header
        short_targets = ["CBM", "DER", "DOR", "GUA", "LLA", "CBL"]
        header_parts = [r"\multicolumn{1}{c}{\bf CATEGORY}"]
        for st in short_targets:
            header_parts.append(rf"\multicolumn{{1}}{{c}}{{\bf {st}}}")
        print(" & ".join(header_parts) + r" \\")
        print(r"\hline \\")

        # Data Rows
        for cat in sorted_categories:
            display_name = cat.replace("_", " ").title()
            row_cells = [f"{display_name:<25}"]
            for target_idx, t_full_name in enumerate(targets):
                val = matrix[cat].get(t_full_name, "--")
                if val == "--":
                    cell_text = "--"
                else:
                    cell_text = f"{val:.1f}\\%"
                    if val > 50:
                        cell_text = f"\\textbf{{{cell_text}}}"
                row_cells.append(cell_text)
            print(" & ".join(row_cells) + r" \\")

        # Average Row
        print(r"\\ \hline \\")
        avg_row = [f"{'Average':<25}"]
        for t_full_name in targets:
            numeric_vals = [
                matrix[cat].get(t_full_name)
                for cat in sorted_categories
                if isinstance(matrix[cat].get(t_full_name), (int, float))
            ]
            if not numeric_vals:
                avg_row.append("--")
            else:
                avg_total = sum(numeric_vals) / len(numeric_vals)
                avg_row.append(f"{avg_total:.1f}\\%")

        print(" & ".join(avg_row) + r" \\")
        print(r"\end{tabular}")
        print(r"\end{subtable}")

        # CHANGE 2: Force vertical stacking with vspace instead of hfill
        if i < 3:  # Don't add space after the last table
            print(r"\\ \vspace{0.8cm}")

    print(r"\end{center}")
    print(r"\end{table*}")


print_vertical_comprehensive_figure()


import numpy as np


def get_bias_summary_data():
    """Filters data for bias categories and computes ASR per attack/target."""
    print("\n% ================= EXTRACTING BIAS SUMMARY DATA =================")

    # Define common bias keywords in XSTest categories
    bias_keywords = [x for x in df["type"].unique() if x.lower().startswith("c")]
    targets = [get_against_name(i).capitalize() for i in range(6)]
    attack_names = [a.__class__.__name__[2:].capitalize() for a in attacks]

    # Structure: { attack_name: { target_name: overall_bias_asr } }
    summary_matrix = {name: {} for name in attack_names}

    for attack_index, attack_obj in enumerate(attacks):
        attack_name = attack_names[attack_index]

        for index in range(6):
            target_name = targets[index]
            base_folder = attack_obj.get_base_folder(index)

            # Use your existing plotting function to get the raw data list
            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            # Filter ddf where the 'type' column contains any bias keyword
            bias_mask = ddf["type"].str.contains("|".join(bias_keywords), case=False, na=False)
            bias_df = ddf[bias_mask]

            if bias_df.empty:
                # Handle cases with no data for these categories
                summary_matrix[attack_name][target_name] = np.nan
                continue

            # Calculate the overall ASR for all bias categories combined
            successful = (bias_df["jailbroken"] == True).sum()
            total = len(bias_df)
            asr = (successful / total * 100) if total > 0 else 0

            summary_matrix[attack_name][target_name] = asr

    return summary_matrix, targets, attack_names


def print_bias_analysis_table():
    print("\n% ================= BIAS-SPECIFIC ANALYSIS TABLE =================")

    # These keywords match the 'type' column strings in XSTest (e.g., 'racial_bias')
    bias_keywords = [x for x in df["type"].unique() if x.lower().startswith("c")]
    targets = [get_against_name(i).capitalize() for i in range(6)]

    for attack_index, attack_obj in enumerate(attacks):
        attack_name = attack_obj.__class__.__name__[2:].capitalize()
        bias_matrix = {}

        for index in range(6):
            target_name = targets[index]
            base_folder = attack_obj.get_base_folder(index)
            # This calls your existing data-loading logic
            datas = plot_base_folder(base_folder, target_name, attack_obj)
            ddf = pd.DataFrame(datas)

            # FILTERING LOGIC: We isolate rows where the 'type' contains our keywords
            bias_df = ddf[ddf["type"].str.contains("|".join(bias_keywords), case=False, na=False)]

            if bias_df.empty:
                continue

            # Calculate ASR (Average Success Rate) for these specific bias rows
            stats = bias_df.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)

            for category, asr in stats.items():
                if category not in bias_matrix:
                    bias_matrix[category] = {}
                bias_matrix[category][target_name] = asr

        if not bias_matrix:
            print(f"% No bias categories found for {attack_name}")
            continue

        # --- GENERATE LATEX ---
        print(r"\begin{table}[h]")
        print(f"\\caption{{Bias-Specific Jailbreak Success Rates (\%): {attack_name}}}")
        print(r"\centering")
        print(r"\begin{tabular}{lcccccc}")
        print(r"\hline")
        # Header with shortened target names
        header = (
            r"\textbf{Bias Type} & "
            + " & ".join([rf"\textbf{{{t[:3].upper()}}}" for t in targets])
            + r" \\"
        )
        print(header)
        print(r"\hline")

        for cat in sorted(bias_matrix.keys()):
            row_str = f"{cat.replace('_', ' ').title():<20}"
            for target in targets:
                val = bias_matrix[cat].get(target, 0.0)
                row_str += f" & {val:.1f}\\%"
            print(row_str + r" \\")

        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")


def get_model_paper_name(index):
    names = ["CBM", "DER", "DOR", "GUA", "LLA", "CBL"]
    return names[index] if index < len(names) else f"Model{index}"


def plot_bias_attacks_summary_figure():
    """Generates a grouped bar chart visualizing bias ASR by attack."""

    # 1. Get the compiled data
    summary_matrix, targets, attack_names = get_bias_summary_data()

    # 2. Set up the figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # 3. Define bar positions and width
    num_attacks = len(attack_names)
    num_targets = len(targets)
    bar_width = 0.20  # Adjust width based on number of attacks

    # Initial X locations for the groups (target models)
    indices = np.arange(num_targets)

    # Define a distinct color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

    # 4. Plot bars for each attack
    for i, attack_name in enumerate(attack_names):
        # Extract ASR values for this specific attack across all targets
        asr_values = [summary_matrix[attack_name].get(t, 0) for t in targets]

        # Calculate the offset for this specific attack's bars within the group
        offset = (i - (num_attacks - 1) / 2) * bar_width

        # Add the bars
        bars = ax.bar(
            indices + offset,
            asr_values,
            bar_width,
            label=attack_name,
            color=colors[i],
            edgecolor="black",
            alpha=0.85,
        )

        # Optional: Add value labels on top of bars > 10%
        for bar in bars:
            height = bar.get_height()
            if height > 10:
                ax.annotate(
                    f"{height:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # 5. Styling and Labels
    ax.set_title(
        "Jailbreak Success Rate (ASR) on Bias-Related Prompts by Attack Method",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("Overall ASR (%)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Target Model", fontsize=20, fontweight="bold")

    # Set X-axis tick labels to the target model names
    ax.set_xticks(indices)

    # Shorten names for better fit if needed (e.g., LlamaInstruct -> LlaIns)
    display_targets = [t[:6] if len(t) > 7 else t for t in targets]
    ax.set_xticklabels(display_targets, rotation=0, fontsize=11)

    # Set Y-axis limits and grid
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add a legend
    ax.legend(title="Attack Method", title_fontsize=12, fontsize=11, loc="upper left")

    # 6. Final Layout and Save
    plt.tight_layout()

    # Save the figure
    output_path = "graphs/bias_attacks_summary_figure.png"
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    plt.savefig(output_path, dpi=300)  # Save high res
    print(f"Figure saved to: {output_path}")

    plt.show()


# 1. Generate the granular LaTeX table for Bias categories
print_bias_analysis_table()

# 2. Generate and save the summary bar chart figure
# plot_bias_attacks_summary_figure()


import seaborn as sns
import matplotlib.pyplot as plt


# def plot_attack_heatmap(attack_index):
#     attack_obj = attacks[attack_index]
#     attack_name = attack_obj.__class__.__name__[2:].capitalize()
#     targets = [get_against_name(i).capitalize() for i in range(6)]

#     # Structure: { Category: { Target: ASR } }
#     matrix_data = []

#     for index in range(6):
#         target_name = targets[index]
#         base_folder = attack_obj.get_base_folder(index)
#         datas = plot_base_folder(base_folder, target_name, attack_obj)
#         ddf = pd.DataFrame(datas)

#         # Calculate ASR per category
#         stats = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)
#         counts = ddf.groupby("type")["jailbroken"].count()
#         empty_types = counts[counts == 0].index.tolist()
#         print(f"{empty_types=}")

#         for category, asr in stats.items():
#             # if not is_empty[category]:
#             if category not in empty_types:
#                 matrix_data.append(
#                     {
#                         "Category": category.replace("_", " ").title().replace("Contrast ", ""),
#                         "Target": get_model_paper_name(index),
#                         "ASR": asr,
#                     }
#                 )

#     # Convert to a pivot table for the heatmap
#     plot_df = pd.DataFrame(matrix_data)
#     heatmap_table = plot_df.pivot(index="Category", columns="Target", values="ASR")

#     # Plotting
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(
#         heatmap_table,
#         annot=True,
#         fmt=".1f",
#         cmap="YlOrRd",
#         cbar_kws={"label": "Success Rate (%)"},
#         annot_kws={"size": 14, "weight": "bold"},
#     )

#     plt.title(f"Vulnerability Heatmap: {attack_name} Attack Success Rate by Category", fontsize=15)
#     plt.tight_layout()
#     plt.savefig(f"graphs/{attack_name.lower()}_heatmap.png")
#     plt.show()


# To run it for the first attack (Crescendo):


import seaborn as sns
import numpy as np


# def plot_dispersion_heatmap(attack_index=0):
#     attack_obj = attacks[attack_index]
#     attack_name = attack_obj.__class__.__name__[2:].capitalize()
#     targets = [get_against_name(i).capitalize() for i in range(6)]

#     all_data = []

#     # 1. Collect ASR per Category per Target
#     for index in range(6):
#         target_name = targets[index]
#         base_folder = attack_obj.get_base_folder(index)
#         datas = plot_base_folder(base_folder, target_name, attack_obj)
#         ddf = pd.DataFrame(datas)

#         # Calculate ASR for each category (type)
#         cat_stats = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)
#         counts = ddf.groupby("type")["jailbroken"].count()
#         empty_types = counts[counts == 0].index.tolist()
#         # Calculate the Global Mean ASR for this specific model
#         model_mean_asr = (ddf["jailbroken"] == True).mean() * 100

#         model_std = 0
#         for category, asr in cat_stats.items():
#             model_std += (asr - model_mean_asr) ** 2

#         model_std = (model_std / len(cat_stats)) ** 0.5

#         for category, asr in cat_stats.items():
#             # Calculate the dispersion (ASR - Mean)
#             if category not in empty_types:
#                 dispersion = asr - model_mean_asr
#                 all_data.append(
#                     {
#                         "Category": category.replace("_", " ").title().replace("Contrast ", ""),
#                         "Target": get_model_paper_name(index),
#                         "Dispersion": (
#                             dispersion / model_std if model_std > 0 else 0
#                         ),  # Standardize by std deviation
#                         "Actual_ASR": asr,
#                     }
#                 )

#     # 2. Pivot the data for the heatmap
#     df_plot = pd.DataFrame(all_data)
#     heatmap_data = df_plot.pivot(index="Category", columns="Target", values="Dispersion")

#     # 3. Plotting
#     plt.figure(figsize=(14, 10))

#     # We use 'RdBu_r' (Red-Blue reversed) so Red = More Vulnerable (+), Blue = Safer (-)
#     # center=0 ensures that the neutral color is exactly at the model's average
#     sns.heatmap(
#         heatmap_data,
#         annot=True,
#         fmt=".1f",
#         cmap="RdBu_r",
#         center=0,
#         cbar_kws={"label": "Deviation from Model Mean ASR (%)"},
#         annot_kws={"size": 14, "weight": "bold"},
#     )

#     plt.title(
#         f"Topic Bias: {attack_name} Attack Dispersion\n(Red: Weak Spot | Blue: Strong Spot relative to average)",
#         fontsize=16,
#     )
#     plt.tight_layout()
#     plt.savefig(f"graphs/{attack_name.lower()}_dispersion_heatmap.png")
#     plt.show()


def plot_attack_heatmap(attack_index):
    attack_obj = attacks[attack_index]
    attack_name = attack_obj.__class__.__name__[2:].capitalize()
    targets = [get_against_name(i).capitalize() for i in range(6)]

    matrix_data = []
    for index in range(6):
        if index == 0:
            continue
        target_name = targets[index]
        base_folder = attack_obj.get_base_folder(index)
        datas = plot_base_folder(base_folder, target_name, attack_obj)
        ddf = pd.DataFrame(datas)

        stats = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)
        counts = ddf.groupby("type")["jailbroken"].count()
        empty_types = counts[counts == 0].index.tolist()

        for category, asr in stats.items():
            if category not in empty_types:
                matrix_data.append(
                    {
                        "Category": category.replace("_", " ").title().replace("Contrast ", ""),
                        "Target": get_model_paper_name(index),
                        "ASR": asr,
                    }
                )

    plot_df = pd.DataFrame(matrix_data)
    heatmap_table = plot_df.pivot(index="Category", columns="Target", values="ASR")

    plt.figure(figsize=(16, 12))

    # 18 BOLD inside the cells
    ax = sns.heatmap(
        heatmap_table,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Success Rate (%)"},
        annot_kws={"size": 18, "weight": "bold"},
    )

    # 18 BOLD on Ticks and Labels
    ax.set_xlabel("Target", fontsize=18, fontweight="bold")
    ax.set_ylabel("Category", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=18, fontweight="bold", rotation=45)
    plt.yticks(fontsize=18, fontweight="bold")

    # Title and Colorbar adjustments
    plt.title(f"Vulnerability Heatmap: {attack_name}", fontsize=22, fontweight="bold", pad=20)
    ax.figure.axes[-1].yaxis.label.set_size(18)

    plt.tight_layout()
    plt.savefig(f"graphs/{attack_name.lower()}_heatmap.png")
    plt.show()


def plot_dispersion_heatmap(attack_index=0):
    attack_obj = attacks[attack_index]
    attack_name = attack_obj.__class__.__name__[2:].capitalize()
    targets = [get_against_name(i).capitalize() for i in range(6)]

    all_data = []
    for index in range(6):
        if index == 0:
            continue
        target_name = targets[index]
        base_folder = attack_obj.get_base_folder(index)
        datas = plot_base_folder(base_folder, target_name, attack_obj)
        ddf = pd.DataFrame(datas)

        cat_stats = ddf.groupby("type")["jailbroken"].apply(lambda x: (x == True).mean() * 100)
        counts = ddf.groupby("type")["jailbroken"].count()
        empty_types = counts[counts == 0].index.tolist()
        model_mean_asr = (ddf["jailbroken"] == True).mean() * 100

        model_std = (sum((asr - model_mean_asr) ** 2 for asr in cat_stats) / len(cat_stats)) ** 0.5

        for category, asr in cat_stats.items():
            if category not in empty_types:
                dispersion = asr - model_mean_asr
                all_data.append(
                    {
                        "Category": category.replace("_", " ").title().replace("Contrast ", ""),
                        "Target": get_model_paper_name(index),
                        "Dispersion": (dispersion / model_std if model_std > 0 else 0),
                        "Actual_ASR": asr,
                    }
                )

    df_plot = pd.DataFrame(all_data)
    heatmap_data = df_plot.pivot(index="Category", columns="Target", values="Dispersion")

    plt.figure(figsize=(18, 12))

    # 18 BOLD inside the cells
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Deviation from Model Mean ASR (%)"},
        annot_kws={"size": 18, "weight": "bold"},
    )

    # 18 BOLD on Ticks and Labels
    ax.set_xlabel("Target", fontsize=18, fontweight="bold")
    ax.set_ylabel("Category", fontsize=18, fontweight="bold")
    plt.xticks(fontsize=18, fontweight="bold", rotation=45)
    plt.yticks(fontsize=18, fontweight="bold")

    plt.title(
        f"Topic Bias: {attack_name} Attack Dispersion", fontsize=22, fontweight="bold", pad=25
    )
    ax.figure.axes[-1].yaxis.label.set_size(18)

    plt.tight_layout()
    plt.savefig(f"graphs/{attack_name.lower()}_dispersion_heatmap.png")
    plt.show()


from typing import Literal


def plot_base_folder_safe(
    base_folder, target_name, attack: Attack, mode: Literal["safe", "unsafe", "all"] = "safe"
):
    from core.attacks.datastore.read import Frame, get_data

    datas = []
    for i, row in df.iterrows():
        if mode == "unsafe":
            if not ("unsafe" in row["id"]):
                continue  # keep only safe
        elif mode == "safe":
            if "unsafe" in row["id"]:
                continue  # keep only safe

        # if not ("How" in row["prompt"]):
        #     continue

        output_folder_name = f"{base_folder}/" + attack.get_subfolder(row["id"])
        info = output_folder_name + "/info.txt"

        frames: list[Frame] = get_data(
            attack.get_subfolder(row["id"]), base_folder, tag_ATTACK=attack.attack_id()
        )
        entry = {"jailbroken": None, "data": {}, **row.to_dict()}
        if os.path.exists(info):
            with open(info, "rt") as f:
                data = f.readline()
                data = data.replace("'", '"')
                data = ast.literal_eval(data)
                if len(list(data.keys())) == 0:
                    entry["jailbroken"] = False
                    entry["data"] = data
                else:
                    first_key = sorted(list(data.keys()))[0]
                    dt = {first_key: data[first_key]}
                    entry["jailbroken"] = True
                    entry["data"] = dt
        else:
            entry["data"] = {}

        refused = 0
        unrefused = 0
        for frame in frames:
            if frame.has_is_refused() and frame.is_response_refused():
                refused += 1
            else:
                unrefused += 1

        entry["refusal_rate"] = (
            refused / (refused + unrefused) if (refused + unrefused) > 0 else np.NaN
        )
        entry["refused"] = refused
        entry["total"] = refused + unrefused
        datas.append(entry)
    # assert len(datas) == 200, f"Expected 250 safe cases, got {len(datas)}"
    return datas


plot_attack_heatmap(0)

# Execute for index 0 (Crescendo)
plot_dispersion_heatmap(0)


def _load_all_safe(
    attack_index: int,
    *,
    only_non_jailbroken: bool = False,
    mode: Literal["safe", "unsafe", "all"] = "safe",
):
    attack_obj = attacks[attack_index]
    frames = []
    for index in range(6):
        if index == 0:
            continue
        target_name = get_against_name(index).capitalize()
        base_folder = attack_obj.get_base_folder(index)
        datas = plot_base_folder_safe(base_folder, target_name, attack_obj, mode=mode)
        if only_non_jailbroken:
            datas = [d for d in datas if d["jailbroken"] == False]
        ddf = pd.DataFrame(datas)
        ddf["target"] = target_name
        ddf["target_short"] = get_model_paper_name(index)
        frames.append(ddf)
    return pd.concat(frames, ignore_index=True)


def plot_refusal_bar_all_attacks(
    *, only_non_jailbroken: bool = False, mode: Literal["safe", "unsafe", "all"] = "safe"
):
    suffix = " (non-jailbroken turns only)" if only_non_jailbroken else " (all turns)"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for attack_index, attack_obj in enumerate(attacks):
        ax = axes[attack_index]
        attack_name = attack_obj.__class__.__name__[2:].capitalize()
        ddf = _load_all_safe(attack_index, only_non_jailbroken=only_non_jailbroken, mode=mode)

        means = ddf.groupby("target_short")["refusal_rate"].mean() * 100
        stds = ddf.groupby("target_short")["refusal_rate"].std() * 100
        order = means.sort_values(ascending=False).index

        print(f"{means=}")

        ax.bar(
            range(len(order)),
            means[order],
            yerr=stds[order],
            capsize=4,
            color="steelblue",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_title(f"{attack_name}{suffix}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Refusal Rate (%)", fontsize=11)
        ax.set_xlabel("Target Model", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        "Refusal Rate on Safe Prompts by Attack & Target Model", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    tag = "nonjb" if only_non_jailbroken else "all"
    plt.savefig(f"graphs/refusal_safe_bar_{tag}.png", dpi=200)
    plt.show()


def plot_refusal_heatmap(attack_index: int, *, only_non_jailbroken: bool = False):
    attack_obj = attacks[attack_index]
    attack_name = attack_obj.__class__.__name__[2:].capitalize()
    suffix = " (non-jailbroken turns only)" if only_non_jailbroken else " (all turns)"

    ddf = _load_all_safe(attack_index, only_non_jailbroken=only_non_jailbroken, mode=mode)

    pivot = ddf.groupby(["type", "target_short"])["refusal_rate"].mean().unstack("target_short")
    col_order = [
        get_model_paper_name(i) for i in range(6) if get_model_paper_name(i) in pivot.columns
    ]
    pivot = pivot[col_order]
    pivot.index = [idx.replace("_", " ").title().replace("Contrast ", "") for idx in pivot.index]

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        cbar_kws={"label": "Refusal Rate (%)"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Target Model", fontsize=16, fontweight="bold")
    ax.set_ylabel("Prompt Category", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14, fontweight="bold", rotation=45)
    plt.yticks(fontsize=13, fontweight="bold")
    plt.title(
        f"Refusal Rate on Safe Prompts – {attack_name}{suffix}",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.figure.axes[-1].yaxis.label.set_size(14)
    plt.tight_layout()
    tag = "nonjb" if only_non_jailbroken else "all"
    plt.savefig(f"graphs/refusal_safe_heatmap_{attack_name.lower()}_{tag}.png", dpi=200)
    plt.show()


def plot_refusal_attacks_summary(*, only_non_jailbroken: bool = False, mode="safe"):
    suffix = " (non-jailbroken turns only)" if only_non_jailbroken else " (all turns)"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    targets_short = [get_model_paper_name(i) for i in range(6)]
    attack_names = [a.__class__.__name__[2:].capitalize() for a in attacks[1:]]

    matrix = {}
    for attack_index, attack_obj in enumerate(attacks):
        if attack_index == 3:
            continue
        ddf = _load_all_safe(attack_index, only_non_jailbroken=only_non_jailbroken, mode=mode)
        means = ddf.groupby("target_short")["refusal_rate"].mean() * 100
        matrix[attack_names[attack_index]] = [means.get(t, np.nan) for t in targets_short]

    num_attacks = len(attacks)
    bar_width = 0.18
    indices = np.arange(len(targets_short))

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (attack_name, vals) in enumerate(matrix.items()):
        offset = (i - (num_attacks - 1) / 2) * bar_width
        bars = ax.bar(
            indices + offset,
            vals,
            bar_width,
            label=attack_name,
            color=colors[i],
            edgecolor="black",
            alpha=0.85,
        )
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 5:
                ax.annotate(
                    f"{h:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_title(
        f"Refusal Rate on {mode.capitalize()} Prompts by Attack Method{suffix}",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Mean Refusal Rate (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Model", fontsize=14, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels(targets_short, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Attack Method", fontsize=11, title_fontsize=12, loc="upper right")

    plt.tight_layout()
    tag = "nonjb" if only_non_jailbroken else "all"
    plt.savefig(f"graphs/refusal_{mode}_summary_{tag}.png", dpi=200)
    plt.show()


def plot_refusal_attacks_summary_safe_unsafe(*, only_non_jailbroken: bool = False):
    suffix = " (non-jailbroken turns only)" if only_non_jailbroken else " (all turns)"
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    targets_short = [get_model_paper_name(i) for i in range(6)][1:]
    attack_names = [a.__class__.__name__[2:].capitalize() for a in attacks if attacks.index(a) != 3]

    # 1. Create extended labels for the X-axis
    # This creates: ['Model A (Safe)', ..., 'Model A (Unsafe)', ...]
    extended_labels = [f"{t}\n(Safe)" for t in targets_short] + [
        f"{t}\n(Unsafe)" for t in targets_short
    ]

    matrix = {}
    # Use a specific list of indices to skip 3 cleanly
    valid_indices = [idx for idx in range(len(attacks)) if idx != 3]

    for attack_index in valid_indices:
        # Get Safe data (6 values)
        ddf_safe = _load_all_safe(
            attack_index, only_non_jailbroken=only_non_jailbroken, mode="safe"
        )
        means_safe = ddf_safe.groupby("target_short")["refusal_rate"].mean() * 100
        safe_vals = [means_safe.get(t, np.nan) for t in targets_short]

        # Get Unsafe data (6 values)
        ddf_unsafe = _load_all_safe(
            attack_index, only_non_jailbroken=only_non_jailbroken, mode="unsafe"
        )
        means_unsafe = ddf_unsafe.groupby("target_short")["refusal_rate"].mean() * 100
        unsafe_vals = [means_unsafe.get(t, np.nan) for t in targets_short]

        # Combine into 12 values total
        matrix[attack_names[valid_indices.index(attack_index)]] = safe_vals + unsafe_vals

    # 2. Update the plotting logic
    indices = np.arange(len(extended_labels))  # This is now length 12
    num_attacks_plotted = len(matrix)
    bar_width = 0.15  # Slightly thinner to fit more bars

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (attack_name, vals) in enumerate(matrix.items()):
        # Centering the group of bars over each of the 12 ticks
        offset = (i - (num_attacks_plotted - 1) / 2) * bar_width
        bars = ax.bar(
            indices + offset,
            vals,
            bar_width,
            label=attack_name,
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.85,
        )
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 5:
                ax.annotate(
                    f"{h:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_title(
        f"Refusal Rate on Safe and Unsafe Prompts by Attack Method{suffix}",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Mean Refusal Rate (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target Model", fontsize=14, fontweight="bold")
    ax.axvline(x=4.5, color="black", linestyle="--", alpha=0.5, linewidth=2)
    ax.text(2.0, 102, "SAFE PROMPTS", ha="center", fontweight="bold", fontsize=12)
    ax.text(7.0, 102, "UNSAFE PROMPTS", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(indices)
    ax.set_xticklabels(extended_labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Attack Method", fontsize=11, title_fontsize=12, loc="upper right")

    plt.tight_layout()
    tag = "nonjb" if only_non_jailbroken else "all"
    plt.savefig(f"graphs/refusal_safe_unsafe_summary_{tag}.png", dpi=200)
    plt.show()


# plot_refusal_bar_all_attacks(only_non_jailbroken=False, mode="safe")
# plot_refusal_attacks_summary(only_non_jailbroken=False, mode="safe")
# plot_refusal_attacks_summary(only_non_jailbroken=False, mode="unsafe")
plot_refusal_attacks_summary_safe_unsafe()
