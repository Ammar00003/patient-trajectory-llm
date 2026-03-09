from pathlib import Path
import math
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY_PATH = PROJECT_ROOT / "data" / "patient_trajectories.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def shorten_event_label(content: str, record_type: str) -> str:
    """Return a short label suitable for plotting."""
    text = str(content).lower()

    if record_type == "medication_change":
        if "furosemide" in text and ("increase" in text or "40" in text):
            return "Furosemide ↑"
        if "spironolactone" in text and ("discontinue" in text or "stop" in text):
            return "Spironolactone stopped"
        if "lactulose" in text and ("change" in text or "30 ml" in text):
            return "Lactulose changed"
        return "Med change"

    if "paracentesis" in text:
        return "Paracentesis"
    if "confusion" in text or "encephalopathy" in text or "hallucination" in text:
        return "Mental status"
    if "distension" in text or "abdominal" in text or "pain" in text:
        return "Pain/distension"
    if "follow" in text or "scheduled" in text or "clinic" in text:
        return "Follow-up"
    if "potassium" in text or "hyperkalemia" in text:
        return "Hyperkalemia"
    if "sodium" in text or "hyponatremia" in text:
        return "Hyponatremia"
    if "albumin" in text:
        return "Albumin"
    if "fluid removed" in text:
        return "Fluid removed"

    words = str(content).split()
    return " ".join(words[:3]) if words else "Event"


def wrap_line(text: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def main():
    df = pd.read_csv(TRAJECTORY_PATH)

    subject_id = input("Enter subject_id to plot: ").strip()

    patient_df = df[df["subject_id"].astype(str) == subject_id].copy()

    if patient_df.empty:
        print(f"No records found for subject_id {subject_id}")
        return

    patient_df = patient_df.sort_values(["admission_index", "event_order"]).reset_index(drop=True)

    admissions = patient_df["admission_index"].drop_duplicates().tolist()
    n_adm = len(admissions)

    # Figure with two panels:
    # top = timeline
    # bottom = detailed event list
    fig_height = max(8, n_adm * 2.4 + 6)
    fig = plt.figure(figsize=(16, fig_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.7], hspace=0.15)

    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    y_gap = 3
    current_y = n_adm * y_gap

    detail_lines = []
    global_counter = 1

    for admission_idx in admissions:
        adm_df = patient_df[patient_df["admission_index"] == admission_idx].copy()

        # Admission line
        ax.hlines(y=current_y, xmin=0, xmax=10, linewidth=2)
        ax.text(
            -0.25, current_y,
            f"Admission {admission_idx}",
            va="center", ha="right", fontsize=11, fontweight="bold"
        )

        n_events = len(adm_df)
        if n_events == 1:
            xs = [5]
        else:
            xs = [1 + (8 * i / max(n_events - 1, 1)) for i in range(n_events)]

        for x, (_, row) in zip(xs, adm_df.iterrows()):
            record_type = row["record_type"]
            marker = "s" if record_type == "medication_change" else "o"

            ax.plot(x, current_y, marker=marker, markersize=7)

            short_label = shorten_event_label(row["content"], record_type)
            ax.text(
                x, current_y + 0.22,
                str(global_counter),
                ha="center", va="bottom", fontsize=7, fontweight="bold"
            )
            ax.text(
                x, current_y - 0.35,
                short_label,
                ha="center", va="top", fontsize=8
            )

            detail_lines.append(
                f"{global_counter}. Admission {admission_idx} | {row['time_phrase']} | {row['content']}"
            )
            global_counter += 1

        current_y -= y_gap

    ax.set_title(f"Patient Trajectory Timeline: subject_id {subject_id}", fontsize=14)
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, n_adm * y_gap + 2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Timeline event',
               markerfacecolor='black', markersize=7),
        Line2D([0], [0], marker='s', color='w', label='Medication change',
               markerfacecolor='black', markersize=7),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=False)

    # Detailed event list panel
    wrapped_lines = [wrap_line(line, width=110) for line in detail_lines]
    details_text = "\n\n".join(wrapped_lines)
    ax_text.text(
        0.0, 1.0,
        details_text,
        ha="left", va="top", fontsize=8, family="monospace"
    )

    # Footer/Branding
    fig.text(
        0.5, 0.01,
        "Powered by LLM-based clinical note extraction",
        ha="center", fontsize=9, style="italic", color="gray"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = OUTPUT_DIR / f"subject_{subject_id}_timeline.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()