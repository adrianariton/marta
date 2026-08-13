import pathlib
import pandas as pd


def calculate_info_rate(directory_path):
    """
    Calculates the percentage of subfolders containing an 'info.txt'
    file that includes the word 'true'.
    """
    base_path = pathlib.Path(directory_path)

    if not base_path.exists():
        return 0.0

    # Get only immediate subdirectories
    subfolders = [f for f in base_path.iterdir() if f.is_dir()]
    total_subfolders = len(subfolders)

    if total_subfolders == 0:
        return 0.0

    true_count = 0
    for folder in subfolders:
        info_file = folder / "info.txt"
        if info_file.exists():
            # Read file and check for "true" (case-insensitive)
            content = info_file.read_text(encoding="utf-8").lower()
            if "true" in content:
                true_count += 1

    return (true_count / total_subfolders) * 100


FITD_folder = "./datasets/fitd_adv_qwen25_vs_{model}__async_batch_bagel_eval_3repl_r10"
GOAT_folder = (
    "./datasets_andu/datasets/goat_adv_qwen25_vs_{model}__async_batch_bagel_eval_3repl_r10"
)
CRESCENDO_folder = (
    "./datasets/crescendo_adv_qwen25_vs_{model}__async_batch_bagel_eval_3repl_3x8_r10"
)

models = ["cbllama", "guardresp", "door", "derta", "llamainstruct"]
shortnames = ["CBL", "GUQ", "DOR", "DER", "LLA"]
fol = {
    "Fitd Attack": FITD_folder,
    "Goat Attack": GOAT_folder,
    "Crescendo": CRESCENDO_folder,
}

# Initialize the DataFrame with Attack Methods as rows and Shortnames as columns
df = pd.DataFrame(index=fol.keys(), columns=shortnames)


def calculate_ratio(attack_key, model):
    folder = fol[attack_key].format(model=model)
    rate = calculate_info_rate(folder)

    # Map model long name to short name for the column header
    short_name = shortnames[models.index(model)]

    # Store in DataFrame
    df.at[attack_key, short_name] = rate


def make_latex(df):
    # Order columns as per your template: DER, DOR, LLA, CBL, GUQ
    column_order = ["DER", "DOR", "LLA", "CBL", "GUQ"]
    # Filter only columns that exist in the dataframe
    existing_columns = [c for c in column_order if c in df.columns]

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Average Simple Safety Jailbreak ASR per Attack Method across Models}",
        "\\label{tab:ss_average_asr_summary}",
        "\\begin{tabular}{l" + "c" * len(existing_columns) + "}",
        "\\hline",
        "\\textbf{Attack Method} & "
        + " & ".join([f"\\textbf{{{col}}}" for col in existing_columns])
        + " \\\\ \\hline",
    ]

    # Iterate through rows in a specific order if desired, otherwise as they appear
    row_order = ["Crescendo", "Goat Attack", "Fitd Attack"]
    for attack in row_order:
        if attack in df.index:
            row_data = [attack]
            for col in existing_columns:
                val = df.at[attack, col]
                row_data.append(f"{val:.2f}\\%")
            latex_lines.append(" & ".join(row_data) + " \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("")
    latex_lines.append("\\vspace{2mm}")
    latex_lines.append("\\small")
    latex_lines.append(
        "\\textbf{Note:} DER: Derta; DOR: DOOR; LLA: Base llama instruct; CBL: CB LLama; GUQ: Question Moderation Guard Qwen over Llama."
    )
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


# --- Execution ---
for attack in fol.keys():
    for model in models:
        calculate_ratio(attack, model)

print(make_latex(df))
