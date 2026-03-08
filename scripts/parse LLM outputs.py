from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "llm_outputs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILENAME_RE = re.compile(
    r"(?P<group>.+?)_subject_(?P<subject_id>\d+)_hadm_(?P<hadm_id>\d+)_note_(?P<note_id>.+?)_(?P<kind>meds|timeline)\.txt$"
)


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.groupdict()


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\d+\.\s*", "", line)   # remove numbered list prefix
    line = re.sub(r"^-\s*", "", line)       # remove bullet prefix
    return line.strip()


def split_sections(text: str):
    """
    Split uppercase heading blocks like:
    MEDS_ON_ADMISSION:
    ...
    """
    lines = text.splitlines()
    sections = {}
    current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # heading like MEDS_ON_ADMISSION: or EVENT_TIMELINE
        if re.fullmatch(r"[A-Z_ ]+:?", line):
            current = line.rstrip(":").strip()
            sections[current] = []
            continue

        if current is not None:
            sections[current].append(raw)

    return sections


def parse_meds_file(path: Path):
    meta = parse_filename(path)
    if meta is None:
        return [], []

    text = path.read_text(encoding="utf-8", errors="replace")
    sections = split_sections(text)

    med_rows = []
    change_rows = []

    section_map = {
        "MEDS_ON_ADMISSION": "admission",
        "MEDICATIONS_ON_ADMISSION": "admission",
        "MEDS_ON_DISCHARGE": "discharge",
        "MEDICATIONS_ON_DISCHARGE": "discharge",
    }

    for section_name, phase in section_map.items():
        if section_name in sections:
            for line in sections[section_name]:
                value = clean_line(line)
                if value and value.upper() != "UNKNOWN":
                    med_rows.append({
                        **meta,
                        "phase": phase,
                        "medication_text": value
                    })

    for change_heading in ["MEDICATION_CHANGES", "MEDS_CHANGES"]:
        if change_heading in sections:
            for line in sections[change_heading]:
                value = clean_line(line)
                if value and value.upper() not in {"UNKNOWN", "NONE"}:
                    change_rows.append({
                        **meta,
                        "change_text": value
                    })

    return med_rows, change_rows


def parse_timeline_file(path: Path):
    meta = parse_filename(path)
    if meta is None:
        return []

    text = path.read_text(encoding="utf-8", errors="replace")
    sections = split_sections(text)

    rows = []
    timeline_lines = sections.get("EVENT_TIMELINE", [])

    event_order = 1
    for line in timeline_lines:
        value = clean_line(line)
        if not value or value.upper() in {"UNKNOWN", "NONE"}:
            continue

        if ":" in value:
            time_phrase, event_text = value.split(":", 1)
            time_phrase = time_phrase.strip()
            event_text = event_text.strip()
        else:
            time_phrase = "UNKNOWN"
            event_text = value

        rows.append({
            **meta,
            "event_order": event_order,
            "time_phrase": time_phrase,
            "event_text": event_text
        })
        event_order += 1

    return rows


def main():
    meds_files = sorted(INPUT_DIR.glob("*_meds.txt"))
    timeline_files = sorted(INPUT_DIR.glob("*_timeline.txt"))

    medication_rows = []
    medication_change_rows = []
    timeline_rows = []

    for path in meds_files:
        meds, changes = parse_meds_file(path)
        medication_rows.extend(meds)
        medication_change_rows.extend(changes)

    for path in timeline_files:
        timeline_rows.extend(parse_timeline_file(path))

    meds_df = pd.DataFrame(medication_rows)
    changes_df = pd.DataFrame(medication_change_rows)
    timeline_df = pd.DataFrame(timeline_rows)

    meds_df.to_csv(OUTPUT_DIR / "medications.csv", index=False)
    changes_df.to_csv(OUTPUT_DIR / "medication_changes.csv", index=False)
    timeline_df.to_csv(OUTPUT_DIR / "timeline_events.csv", index=False)

    print("Parsing complete.")
    print(f"Medication rows: {len(meds_df)}")
    print(f"Medication change rows: {len(changes_df)}")
    print(f"Timeline event rows: {len(timeline_df)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()