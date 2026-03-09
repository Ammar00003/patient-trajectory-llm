from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = PROJECT_ROOT / "data" / "selected_cohort.csv"
TIMELINE_PATH = PROJECT_ROOT / "data" / "parsed_outputs" / "timeline_events.csv"
CHANGES_PATH = PROJECT_ROOT / "data" / "parsed_outputs" / "medication_changes.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "patient_trajectories.csv"


def main():
    cohort = pd.read_csv(COHORT_PATH)
    timeline = pd.read_csv(TIMELINE_PATH)
    changes = pd.read_csv(CHANGES_PATH)

    # keep one row per admission for ordering
    admissions = (
        cohort[["group", "subject_id", "hadm_id", "note_id", "storetime"]]
        .drop_duplicates()
        .copy()
    )

    admissions["storetime"] = pd.to_datetime(admissions["storetime"], errors="coerce")
    admissions = admissions.sort_values(["subject_id", "storetime", "hadm_id"])

    # assign admission order per patient
    admissions["admission_index"] = admissions.groupby("subject_id").cumcount() + 1

    # merge timeline events with admission order
    timeline = timeline.merge(
        admissions[["group", "subject_id", "hadm_id", "note_id", "storetime", "admission_index"]],
        on=["group", "subject_id", "hadm_id", "note_id"],
        how="left"
    )

    timeline["record_type"] = "timeline_event"
    timeline["content"] = timeline["event_text"]

    timeline_out = timeline[[
        "group", "subject_id", "hadm_id", "note_id",
        "storetime", "admission_index",
        "record_type", "event_order", "time_phrase", "content"
    ]].copy()

    # turn medication changes into trajectory records too
    if not changes.empty:
        changes = changes.merge(
            admissions[["group", "subject_id", "hadm_id", "note_id", "storetime", "admission_index"]],
            on=["group", "subject_id", "hadm_id", "note_id"],
            how="left"
        )

        changes["record_type"] = "medication_change"
        changes["time_phrase"] = "during hospital stay"
        changes["content"] = changes["change_text"]

        # assign sequential order to medication changes after existing timeline events
        # we do this by adding the number of timeline events in that admission
        max_orders = (
            timeline_out.groupby(["subject_id", "admission_index"])["event_order"]
            .max()
            .reset_index()
            .rename(columns={"event_order": "max_timeline_order"})
        )
        
        changes = changes.merge(max_orders, on=["subject_id", "admission_index"], how="left")
        changes["max_timeline_order"] = changes["max_timeline_order"].fillna(0).astype(int)
        
        # within each admission, order medication changes sequentially
        changes["med_order"] = changes.groupby(["subject_id", "admission_index"]).cumcount() + 1
        changes["event_order"] = (changes["max_timeline_order"] + changes["med_order"]).astype(int)

        changes_out = changes[[
            "group", "subject_id", "hadm_id", "note_id",
            "storetime", "admission_index",
            "record_type", "event_order", "time_phrase", "content"
        ]].copy()
    else:
        changes_out = pd.DataFrame(columns=timeline_out.columns)

    trajectories = pd.concat([timeline_out, changes_out], ignore_index=True)
    trajectories["event_order"] = trajectories["event_order"].astype(int)
    trajectories = trajectories.sort_values(
        ["group", "subject_id", "admission_index", "event_order"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(OUTPUT_PATH, index=False)

    print("Patient trajectory dataset created.")
    print(f"Rows: {len(trajectories)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()