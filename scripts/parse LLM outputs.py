from pathlib import Path
import re
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.cleaning_utils import clean_gemma_output

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
    line = re.sub(r"^-\s*", "", line)       # remove dash bullet prefix
    line = re.sub(r"^\*\s*", "", line)      # remove asterisk bullet prefix
    return line.strip()


def split_sections(text: str):
    """
    Split uppercase heading blocks like:
    MEDS_ON_ADMISSION:
    or
    MEDS_ON_ADMISSION
    """
    lines = text.splitlines()
    sections = {}
    current = None

    # Known section headers to look for
    known_headers = {
        "MEDS_ON_ADMISSION", "MEDICATIONS_ON_ADMISSION",
        "MEDS_ON_DISCHARGE", "MEDICATIONS_ON_DISCHARGE",
        "MEDICATION_CHANGES", "MEDS_CHANGES",
        "EVENT_TIMELINE"
    }

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Check if line matches a known header (with or without colon)
        line_normalized = line.rstrip(":").strip().upper()
        if line_normalized in known_headers:
            current = line_normalized
            sections[current] = []
            continue

        # Fallback: heading like MEDS_ON_ADMISSION: or EVENT_TIMELINE
        if re.fullmatch(r"[A-Z_ ]+:?", line) and len(line) > 3:
            current = line.rstrip(":").strip().upper()
            if current not in sections:
                sections[current] = []
            continue

        if current is not None:
            sections[current].append(raw)

    return sections


def parse_meds_file(path: Path, clean_gemma=False):
    meta = parse_filename(path)
    if meta is None:
        return [], []

    text = path.read_text(encoding="utf-8", errors="replace")

    # Clean Gemma output if requested
    if clean_gemma:
        text = clean_gemma_output(text)

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


def parse_timeline_file(path: Path, clean_gemma=False):
    meta = parse_filename(path)
    if meta is None:
        return []

    text = path.read_text(encoding="utf-8", errors="replace")

    # Clean Gemma output if requested
    if clean_gemma:
        text = clean_gemma_output(text)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--clean-gemma", action="store_true", help="Clean Gemma 4 output (remove thinking process and ANSI codes)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meds_files = sorted(input_dir.glob("*_meds.txt"))
    timeline_files = sorted(input_dir.glob("*_timeline.txt"))

    medication_rows = []
    medication_change_rows = []
    timeline_rows = []

    for path in meds_files:
        meds, changes = parse_meds_file(path, clean_gemma=args.clean_gemma)
        medication_rows.extend(meds)
        medication_change_rows.extend(changes)

    for path in timeline_files:
        timeline_rows.extend(parse_timeline_file(path, clean_gemma=args.clean_gemma))

    meds_df = pd.DataFrame(medication_rows)
    changes_df = pd.DataFrame(medication_change_rows)
    timeline_df = pd.DataFrame(timeline_rows)

    meds_df.to_csv(output_dir / "medications.csv", index=False)
    changes_df.to_csv(output_dir / "medication_changes.csv", index=False)
    timeline_df.to_csv(output_dir / "timeline_events.csv", index=False)

    print("Parsing complete.")
    print(f"Medication rows: {len(meds_df)}")
    print(f"Medication change rows: {len(changes_df)}")
    print(f"Timeline event rows: {len(timeline_df)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()