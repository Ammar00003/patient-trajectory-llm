from pathlib import Path
import pandas as pd
from utils.visualization_utils import build_patient_timeline_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY_PATH = PROJECT_ROOT / "data" / "patient_trajectories.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not TRAJECTORY_PATH.exists():
        print(f"Error: Data file not found at {TRAJECTORY_PATH}")
        return

    df = pd.read_csv(TRAJECTORY_PATH)

    subject_id = input("Enter subject_id to plot: ").strip()
    if not subject_id:
        print("Error: No subject_id provided.")
        return

    patient_df = df[df["subject_id"].astype(str) == subject_id].copy()

    if patient_df.empty:
        print(f"No records found for subject_id {subject_id}")
        return

    print(f"Generating professional timeline for Subject {subject_id}...")
    fig = build_patient_timeline_figure(patient_df, subject_id)

    html_path = OUTPUT_DIR / f"subject_{subject_id}_timeline_interactive.html"
    fig.write_html(html_path)
    fig.show()

    print(f"Saved interactive figure to: {html_path}")


if __name__ == "__main__":
    main()