from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = PROJECT_ROOT / "data" / "selected_cohort.csv"
TIMELINE_PATH = PROJECT_ROOT / "data" / "parsed_outputs" / "timeline_events.csv"
CHANGES_PATH = PROJECT_ROOT / "data" / "inferred_outputs" / "inferred_medication_changes.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "patient_trajectories.csv"

def get_time_phrase_order(time_phrase):
    """
    Assign ordering value to time phrases.
    Standard phases: on admission (1) -> during hospital stay (2) -> at discharge (3)
    Non-standard phrases get 0 (sorted to front)
    """
    if pd.isna(time_phrase):
        return 0

    phrase = str(time_phrase).strip().lower()

    # Standard time phrases in chronological order
    if phrase in ['on admission', 'at admission', 'on presentation', 'upon arrival']:
        return 1
    elif phrase in ['during hospital stay', 'during stay', 'hospital course']:
        return 2
    elif phrase in ['at discharge', 'on discharge']:
        return 3
    else:
        # Non-standard phrases (e.g., "3 days ago", "POD2") go to front
        return 0

def format_med_change(row):
    ctype = str(row.get("change_type", "")).strip().upper()

    medication = row.get("medication_name", "UNKNOWN")
    before = row.get("before")
    after = row.get("after")

    # Handle pandas NaN
    if pd.isna(before):
        before = None
    if pd.isna(after):
        after = None
    if pd.isna(medication):
        medication = "UNKNOWN"

    if ctype == "ADDED":
        return f"Added: {after or medication}"

    if ctype == "DISCONTINUED":
        return f"Discontinued: {before or medication}"

    if ctype == "DOSE_CHANGED":
        if before and after:
            return f"Dose changed: {before} -> {after}"
        return f"Dose changed: {medication}"

    if ctype == "FREQUENCY_CHANGED":
        if before and after:
            return f"Frequency changed: {before} -> {after}"
        return f"Frequency changed: {medication}"

    if ctype == "ROUTE_CHANGED":
        if before and after:
            return f"Route changed: {before} -> {after}"
        return f"Route changed: {medication}"

    if ctype == "MODIFIED":
        if before and after:
            return f"Modified: {before} -> {after}"
        return f"Modified: {medication}"

    return f"{ctype}: {medication}"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=str, default=str(TIMELINE_PATH))
    parser.add_argument("--changes", type=str, default=str(CHANGES_PATH))
    parser.add_argument("--counts",  type=str, default=None,
                        help="Path to medication_counts.csv. Defaults to medication_counts.csv in the same folder as --changes.")
    parser.add_argument("--output",  type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    timeline_path = Path(args.timeline)
    changes_path  = Path(args.changes)
    counts_path   = Path(args.counts) if args.counts else changes_path.parent / "medication_counts.csv"
    output_path   = Path(args.output)

    cohort = pd.read_csv(COHORT_PATH)
    timeline = pd.read_csv(timeline_path)
    changes = pd.read_csv(changes_path) if changes_path.exists() else pd.DataFrame()

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

    # Add time phrase ordering for proper sorting
    timeline["time_order"] = timeline["time_phrase"].apply(get_time_phrase_order)

    timeline_out = timeline[[
        "group", "subject_id", "hadm_id", "note_id",
        "storetime", "admission_index",
        "record_type", "event_order", "time_phrase", "time_order", "content"
    ]].copy()

    # turn medication changes into trajectory records too
    if not changes.empty:
        changes = changes.merge(
            admissions[["group", "subject_id", "hadm_id", "note_id", "storetime", "admission_index"]],
            on=["group", "subject_id", "hadm_id", "note_id"],
            how="left"
        )

        changes["record_type"] = "medication_change"
        changes["time_phrase"] = ""  # Remove time description from medication changes
        changes["time_order"] = 4  # Place medication changes after all timeline events (max standard is 3)
        changes["content"] = changes.apply(format_med_change, axis=1)

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
            "record_type", "event_order", "time_phrase", "time_order", "content"
        ]].copy()
    else:
        changes_out = pd.DataFrame(columns=timeline_out.columns)

    trajectories = pd.concat([timeline_out, changes_out], ignore_index=True)
    trajectories["event_order"] = trajectories["event_order"].astype(int)
    trajectories["time_order"] = trajectories["time_order"].astype(int)

    # Sort by: patient -> admission -> time_order (0=non-standard first, 1=admission, 2=during, 3=discharge) -> event_order
    trajectories = trajectories.sort_values(
        ["group", "subject_id", "admission_index", "time_order", "event_order"]
    )

            # Reassign event_order to reflect sorted position so the timeline visualisation preserves this order
    trajectories["event_order"] = (
        trajectories.groupby(["subject_id", "admission_index"]).cumcount() + 1
    ).astype(int)

    # Drop time_order column before saving (only needed for sorting)
    trajectories = trajectories.drop(columns=["time_order"])

    # Merge medication counts (adm_med_count, dis_med_count) per admission
    if counts_path.exists():
        counts = pd.read_csv(counts_path)
        counts = counts.merge(
            admissions[["group", "subject_id", "hadm_id", "note_id", "admission_index"]],
            on=["group", "subject_id", "hadm_id", "note_id"],
            how="left"
        )
        trajectories = trajectories.merge(
            counts[["subject_id", "admission_index", "adm_med_count", "dis_med_count"]],
            on=["subject_id", "admission_index"],
            how="left"
        )
    else:
        trajectories["adm_med_count"] = None
        trajectories["dis_med_count"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output_path, index=False)

    print("Patient trajectory dataset created.")
    print(f"Rows: {len(trajectories)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()