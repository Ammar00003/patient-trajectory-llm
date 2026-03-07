from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "discharge.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "selected_cohort.csv"

SINGLE_TARGET = 25
MULTI_TARGET = 25


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("Total rows before note_type filtering:", len(df))

    # Keep discharge summaries only
    df = df[df["note_type"].astype(str).str.strip().str.upper() == "DS"]

    print("Total rows after note_type filtering:", len(df))

    # Remove rows missing critical fields
    df = df.dropna(subset=["subject_id", "hadm_id", "text"])

    # Convert types
    df["subject_id"] = df["subject_id"].astype(str)
    df["hadm_id"] = df["hadm_id"].astype(str)
    df["note_seq"] = pd.to_numeric(df["note_seq"], errors="coerce")
    df["storetime"] = pd.to_datetime(df["storetime"], errors="coerce")
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

    # Prefer storetime, fallback to charttime
    df["sort_time"] = df["storetime"].fillna(df["charttime"])

    # Keep one discharge summary per admission
    df = df.sort_values(["subject_id", "hadm_id", "sort_time", "note_seq"])
    df = df.drop_duplicates(subset=["subject_id", "hadm_id"], keep="last")

    # Count admissions per patient
    admission_counts = (
        df.groupby("subject_id")["hadm_id"]
        .nunique()
        .reset_index(name="num_admissions")
    )

    single_patients = admission_counts[admission_counts["num_admissions"] == 1].copy()
    multi_patients = admission_counts[admission_counts["num_admissions"] >= 3].copy()

    # Select target cohort
    selected_single = single_patients.head(SINGLE_TARGET).copy()
    selected_multi = multi_patients.head(MULTI_TARGET).copy()

    selected_single["group"] = "single"
    selected_multi["group"] = "multi_3plus"

    selected = pd.concat([selected_single, selected_multi], ignore_index=True)

    # Join back to admission-level notes
    cohort = df.merge(selected[["subject_id", "group"]], on="subject_id")
    cohort = cohort.sort_values(["group", "subject_id", "sort_time"])

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT_PATH, index=False)

    print("\nCohort selection complete")
    print("Single admission patients:", len(selected_single))
    print("Multi-admission patients:", len(selected_multi))
    print("Total admissions selected:", len(cohort))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()