from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = PROJECT_ROOT / "data" / "selected_cohort.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sectioned_notes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Primary headings and common variants seen in discharge summaries
SECTION_PATTERNS = {
    "chief_complaint": [
        r"chief complaint[:\s]*",
    ],
    "history_present_illness": [
        r"history of present illness[:\s]*",
        r"hpi[:\s]*",
    ],
    "major_procedure": [
        r"major surgical or invasive procedure[:\s]*",
        r"major procedure[:\s]*",
    ],
    "past_medical_history": [
        r"past medical history[:\s]*",
    ],
    "physical_exam": [
        r"physical exam[:\s]*",
        r"discharge[:\s]*physical examination[:\s]*",
        r"discharge physical exam[:\s]*",
    ],
    "pertinent_results": [
        r"pertinent results[:\s]*",
        r"pertinent laboratory values[:\s]*",
        r"results[:\s]*",
    ],
    "brief_hospital_course": [
        r"brief hospital course[:\s]*",
        r"hospital course[:\s]*",
    ],
    "medications_on_admission": [
        r"medications on admission[:\s]*",
        r"preadmission medications[:\s]*",
        r"preadmission medication list[:\s]*",
    ],
    "discharge_medications": [
        r"discharge medications[:\s]*",
    ],
    "discharge_disposition": [
        r"discharge disposition[:\s]*",
    ],
    "discharge_diagnosis": [
        r"discharge diagnosis[:\s]*",
    ],
    "discharge_condition": [
        r"discharge condition[:\s]*",
    ],
    "discharge_instructions": [
        r"discharge instructions[:\s]*",
        r"followup instructions[:\s]*",
        r"follow-up instructions[:\s]*",
    ],
}


def normalise_text(text: str) -> str:
    """
    Light cleanup only.
    Keeps the note essentially raw while making regex matching more reliable.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # preserve line breaks but collapse excessive blank space
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def find_all_heading_matches(text: str):
    """
    Return all section heading matches as:
    [(start_index, end_index, canonical_section_name, matched_heading_text), ...]
    """
    matches = []

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            regex = re.compile(pattern, flags=re.IGNORECASE)
            for m in regex.finditer(text):
                matches.append(
                    (m.start(), m.end(), section_name, m.group(0))
                )

    # sort by note position
    matches.sort(key=lambda x: x[0])

    # de-duplicate very close / overlapping matches
    deduped = []
    last_start = -1
    last_end = -1
    for item in matches:
        start, end, section_name, matched = item
        if start < last_end and start == last_start:
            continue
        deduped.append(item)
        last_start, last_end = start, end

    return deduped


def split_sections(text: str) -> dict:
    """
    Split note into sections based on heading matches.
    Returns dict: {section_name: extracted_text}
    If a section appears multiple times, concatenates them.
    """
    text = normalise_text(text)
    matches = find_all_heading_matches(text)

    if not matches:
        return {}

    sections = {}

    for i, (start, end, section_name, matched) in enumerate(matches):
        content_start = end
        content_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if section_name in sections:
            sections[section_name] += "\n\n" + content
        else:
            sections[section_name] = content

    return sections


def build_meds_text(sections: dict) -> str:
    """
    Build medication-focused text from relevant sections.
    """
    ordered_keys = [
        "medications_on_admission",
        "discharge_medications",
        "brief_hospital_course",
        "discharge_instructions",
    ]

    blocks = []
    for key in ordered_keys:
        if key in sections and sections[key].strip():
            blocks.append(f"{key.upper()}:\n{sections[key].strip()}")

    return "\n\n".join(blocks).strip()


def build_timeline_text(sections: dict) -> str:
    """
    Build timeline-focused text from relevant sections.
    """
    ordered_keys = [
        "chief_complaint",
        "history_present_illness",
        "major_procedure",
        "brief_hospital_course",
        "discharge_instructions",
    ]

    blocks = []
    for key in ordered_keys:
        if key in sections and sections[key].strip():
            blocks.append(f"{key.upper()}:\n{sections[key].strip()}")

    return "\n\n".join(blocks).strip()


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    print("Loading selected cohort...")
    df = pd.read_csv(COHORT_PATH)

    total = len(df)
    extracted = 0
    no_sections = 0

    # Optional test mode
    #df = df.head(3)

    for i, row in df.iterrows():
        subject_id = str(row["subject_id"])
        hadm_id = str(row["hadm_id"])
        note_id = str(row["note_id"])
        group = str(row["group"])
        note_text = str(row["text"])

        base_name = f"{group}_subject_{subject_id}_hadm_{hadm_id}_note_{note_id}"

        sections = split_sections(note_text)

        meds_text = build_meds_text(sections)
        timeline_text = build_timeline_text(sections)

        if not meds_text and not timeline_text:
            no_sections += 1
            print(f"[WARN {i+1}/{total}] No sections found for {base_name}")
            continue

        save_text(OUTPUT_DIR / f"{base_name}_meds_input.txt", meds_text)
        save_text(OUTPUT_DIR / f"{base_name}_timeline_input.txt", timeline_text)

        extracted += 1
        print(f"[DONE {i+1}/{total}] {base_name}")

    print("\nSection extraction complete.")
    print(f"Processed: {extracted}")
    print(f"No sections found: {no_sections}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()