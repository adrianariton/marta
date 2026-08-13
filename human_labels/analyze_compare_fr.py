import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import os


def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def analyze_goal_revealed_agreement(
    file1_path, file2_path, file3_path, output_dir, judge1_name="Judge1", judge2_name="Judge2", args=None
):
    """Compare two human judges on goal_revealed ONLY with agreement metrics."""

    # Read both files
    data1 = read_jsonl(file1_path)
    data2 = read_jsonl(file2_path)
    data3 = read_jsonl(file3_path)
    

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    
    # print(df3[args.ai].head())
    # print(pd.json_normalize(df3[args.ai].map(json.loads, na_action='ignore')).head())

    # Extract goal_revealed only
    human1 = pd.json_normalize(df1["human_eval"])["goal_revealed"].astype(int)
    human2 = pd.json_normalize(df2["human_eval"])["goal_revealed"].astype(int)
    # ai1 = pd.json_normalize(df1["ai_eval"])["goal_revealed"].astype(int)
    # ai2 = pd.json_normalize(df2["ai_eval"])["goal_revealed"].astype(int)
    try:
        ai3 = pd.json_normalize(df3[args.ai].fillna("{}").map(json.loads, na_action='ignore'))["harmful"].fillna(0).astype(int)
    except Exception as e:
        ai3 = pd.json_normalize(df3[args.ai])["goal_revealed"].astype(int)
        
    print(ai3.head())
    # assert ai1.equals(ai2)
    n = len(human1)

    # Individual judge metrics (LLM-as-a-Judge bias correction)
    def judge_metrics(human, ai):
        assert len(human) == len(ai)
        p_human = human.mean()
        p_ai = ai.mean()

        tp = ((ai == 1) & (human == 1)).sum()
        tn = ((ai == 0) & (human == 0)).sum()
        fp = ((ai == 1) & (human == 0)).sum()
        fn = ((ai == 0) & (human == 1)).sum()

        q1 = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        q0 = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1 = 2 * precision * q1 / (precision + q1) if (precision + q1) > 0 else 0.0

        denominator = q1 - (1 - q0)
        theta_adj = (
            max(0, min(1, (p_ai - (1 - q0)) / denominator)) if abs(denominator) > 1e-6 else p_ai
        )

        se = np.sqrt((p_ai * (1 - p_ai) / n) + (p_human * (1 - p_human) / n))
        z_score = (p_ai - p_human) / se if se > 0 else np.nan

        corr = human.corr(ai)

        return {
            "p_human": p_human,
            "p_ai": p_ai,
            "theta_adj": theta_adj,
            "q1": q1,
            "q0": q0,
            "q0_plus_q1": q0 + q1,
            "precision": precision,
            "f1": f1,
            "z_score": z_score,
            "corr": corr,
        }

    metrics1 = judge_metrics(human1, ai3)
    metrics2 = judge_metrics(human2, ai3)
    consensus_mask = human1 == human2
    human_consensus = human1[consensus_mask]
    ai_consensus = ai3[consensus_mask]  # or ai2, they should be the same

    # Now calculate how the AI performs ONLY on the samples humans agreed on
    ai_performance_vs_truth = judge_metrics(human_consensus, ai_consensus)
    print(f"AI Performance on Consensus Data: {ai_performance_vs_truth['theta_adj']:.1%}")
    # AI performance on consensus data (detailed)
    cons_metrics = ai_performance_vs_truth
    n_consensus = consensus_mask.sum()

    print(f"\n{'='*70}")
    print(f"🤖 AI PERFORMANCE (n_consensus={n_consensus}, n_total={n})")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'vs J1':<15} {'vs J2':<15} {'vs Consensus':<15}")
    print(f"{'-'*70}")
    print(
        f"{'precision':<20} {metrics1['precision']:.3f}{'':>9} {metrics2['precision']:.3f}{'':>9} {cons_metrics['precision']:.3f}"
    )
    print(
        f"{'F1 score':<20} {metrics1['f1']:.3f}{'':>9} {metrics2['f1']:.3f}{'':>9} {cons_metrics['f1']:.3f}"
    )
    print(
        f"{'θ_adj %':<20} {metrics1['theta_adj']:.1%}{'':>9} {metrics2['theta_adj']:.1%}{'':>9} {cons_metrics['theta_adj']:.1%}"
    )
    print(
        f"{'q₁ (sensitivity)':<20} {metrics1['q1']:.3f}{'':>9} {metrics2['q1']:.3f}{'':>9} {cons_metrics['q1']:.3f}"
    )
    print(
        f"{'q₀ (specificity)':<20} {metrics1['q0']:.3f}{'':>9} {metrics2['q0']:.3f}{'':>9} {cons_metrics['q0']:.3f}"
    )
    print(
        f"{'q₀+q₁':<20} {metrics1['q0_plus_q1']:.3f}{'':>9} {metrics2['q0_plus_q1']:.3f}{'':>9} {cons_metrics['q0_plus_q1']:.3f}"
    )
    print(
        f"{'Z-score':<20} {metrics1['z_score']:.2f}{'':>11} {metrics2['z_score']:.2f}{'':>11} {cons_metrics['z_score']:.2f}"
    )
    print(
        f"{'corr (r)':<20} {metrics1['corr']:.3f}{'':>9} {metrics2['corr']:.3f}{'':>9} {cons_metrics['corr']:.3f}"
    )
    print(f"{'-'*70}")
    print(f"{'Pooled θ_adj':<20} {((metrics1['theta_adj'] + metrics2['theta_adj'])/2):.1%}")
    print(f"{'Coverage':<20} {n_consensus}/{n} ({n_consensus/n:.0%}) samples with human consensus")

    # Inter-judge agreement
    judge_agreement = (human1 == human2).mean()
    p_e = (human1.mean() * human2.mean()) + ((1 - human1.mean()) * (1 - human2.mean()))
    kappa = (judge_agreement - p_e) / (1 - p_e)
    # kappa = (judge_agreement - 0.5) / (1 - 0.5)  # Simple Cohen's kappa for balanced priors
    judge_corr = human1.corr(human2)

    # Print results
    print(f"\n{'='*70}")
    print(f"🎯 GOAL_REVEALED: JUDGE AGREEMENT ANALYSIS (n={n})")
    print(f"{'='*70}")
    print(f"{'JUDGE 1 (' + judge1_name + ')':<25} | JUDGE 2 (" + judge2_name + ")")
    print(f"{'-'*70}")
    print(f"{'Human %':<12} {metrics1['p_human']:.1%} {'|'*10} {metrics2['p_human']:.1%}")
    print(f"{'AI % (raw)':<12} {metrics1['p_ai']:.1%} {'|'*10} {metrics2['p_ai']:.1%}")
    print(f"{'θ_adj %':<12} {metrics1['theta_adj']:.1%} {'|'*10} {metrics2['theta_adj']:.1%}")
    print(f"{'q₀+q₁':<12} {metrics1['q0_plus_q1']:.3f} {'|'*10} {metrics2['q0_plus_q1']:.3f}")
    print(f"{'Z-score':<12} {metrics1['z_score']:.2f} {'|'*10} {metrics2['z_score']:.2f}")
    print(f"{'AI corr (r)':<12} {metrics1['corr']:.3f} {'|'*10} {metrics2['corr']:.3f}")
    print(f"{'-'*70}")
    print(f"{'Judge Agreement':<12} {judge_agreement:.1%}")
    print(f"{'Judge Kappa':<12} {kappa:.3f}")
    print(f"{'Judge corr (r)':<12} {judge_corr:.3f}")
    print(f"{'Pooled θ_adj':<12} {((metrics1['theta_adj'] + metrics2['theta_adj'])/2):.1%}")

    # Interpretation
    print(f"\n📋 INTERPRETATION:")
    print(
        f"• Judge agreement: {'✅ HIGH' if judge_agreement > 0.7 else '⚠️  MODERATE' if judge_agreement > 0.5 else '❌ LOW'} ({judge_agreement:.1%})"
    )
    print(
        f"• Both judges {'✅ STRONG' if min(metrics1['q0_plus_q1'], metrics2['q0_plus_q1']) > 1.3 else '⚠️  OK'} (q₀+q₁ > 1.3)"
    )
    print(
        f"• {'✅ NO' if abs(metrics1['z_score']) < 2 and abs(metrics2['z_score']) < 2 else '⚠️  YES'} extreme raw bias detected"
    )
    print(
        f"• Final pooled estimate: {((metrics1['theta_adj'] + metrics2['theta_adj'])/2):.1%} [range: {min(metrics1['theta_adj'], metrics2['theta_adj']):.0%}-{max(metrics1['theta_adj'], metrics2['theta_adj']):.0%}]"
    )

    # Plot: Side-by-side confusion matrices + agreement heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Judge 1 confusion matrix
    cm1 = pd.crosstab(human1, ai3)
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(
        f'{judge1_name}\nθ_adj={metrics1["theta_adj"]:.0%} | Z={metrics1["z_score"]:.1f}'
    )
    axes[0].set_xlabel("AI")
    axes[0].set_ylabel("Human")

    # Judge 2 confusion matrix
    cm2 = pd.crosstab(human2, ai3)
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False)
    axes[1].set_title(
        f'{judge2_name}\nθ_adj={metrics2["theta_adj"]:.0%} | Z={metrics2["z_score"]:.1f}'
    )
    axes[1].set_xlabel("AI")
    axes[1].set_ylabel("Human")

    # Judge agreement heatmap
    agreement_table = pd.crosstab(human1, human2, rownames=["Judge1"], colnames=["Judge2"])
    sns.heatmap(
        agreement_table, annot=True, fmt="d", cmap="Greens", ax=axes[2], cbar_kws={"label": "Count"}
    )
    axes[2].set_title(f"Judge Agreement\n{judge_agreement:.0%} agreement")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/judge_comparison_goal_revealed.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "judge1": metrics1,
        "judge2": metrics2,
        "agreement": judge_agreement,
        "kappa": kappa,
        "pooled_theta": (metrics1["theta_adj"] + metrics2["theta_adj"]) / 2,
    }


# Usage - replace with your actual file paths
parser = argparse.ArgumentParser()
parser.add_argument("file1", help="First judge's JSONL file")
parser.add_argument("file2", help="Second judge's JSONL file")
parser.add_argument("file3", help="AI's JSONL file")
parser.add_argument("--judge1", default="You", help="Name for judge 1")
parser.add_argument("--judge2", default="Brother", help="Name for judge 2")
parser.add_argument("--ai", default="ai_eval", help="Name for AI")
parser.add_argument("--out", default="./plots", help="Output directory")
parser.add_argument("-s", "--showonly", action="store_true")
args = parser.parse_args()

output_dir = f"{args.out}/judge_comparison"
os.makedirs(output_dir, exist_ok=True)

# Run analysis
results = analyze_goal_revealed_agreement(
    args.file1, args.file2, args.file3, output_dir, args.judge1, args.judge2, args
)

print("\n✅ Analysis complete! Check plots and pooled estimate above.")
# python3 analyze_compare.py AdiQVB.jsonl AnduQVB.jsonl --judge1 "J1" --judge2 "J2"


# Judges are: 'ai_eval' - which is bagel, 'eval_PEC7_phi-4-AWQ', 'eval_PEC7_Mixtral-8x22B-Instruct-v0.1-AWQ', 'eval_PEC7_Qwen2.5-32B-Instruct-AWQ', 'eval_PEC7_Qwen2.5-72B-Instruct-AWQ'
