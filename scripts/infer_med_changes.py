import re
from typing import Optional
import sys

import pandas as pd
from pathlib import Path
# from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.cleaning_utils import clean_gemma_output

# We'll default to llm_outputs as requested for the original LLM
INPUT_DIR = PROJECT_ROOT / "llm_outputs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "inferred_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILENAME_RE = re.compile(
    r"(?P<group>.+?)_subject_(?P<subject_id>\d+)_hadm_(?P<hadm_id>\d+)_note_(?P<note_id>.+?)_meds\.txt$"
)

def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.groupdict()

def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\d+\.\s*", "", line)   # remove numbered list prefix
    line = re.sub(r"^-\s*", "", line, flags=re.IGNORECASE)       # remove bullet prefix
    line = re.sub(r"^\*\s*", "", line)      # remove asterisk bullet
    return line.strip()

def split_sections(text: str):
    lines = text.splitlines()
    sections = {}
    current = None

    # Common section headers
    HEADERS = ["MEDS_ON_ADMISSION", "MEDICATIONS_ON_ADMISSION", "MEDS_ON_DISCHARGE", "MEDICATIONS_ON_DISCHARGE", "MEDICATION_CHANGES", "MEDS_CHANGES"]

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Check if line is a header
        upper_line = line.rstrip(":").strip().upper()
        if upper_line in HEADERS:
            current = upper_line
            sections[current] = []
            continue
        
        # Fallback for headers that might be formatted slightly differently but still uppercase
        if re.fullmatch(r"[A-Z_ ]+:?", line) and len(line) > 3:
             test_header = line.rstrip(":").strip().upper()
             if test_header in HEADERS or any(h in test_header for h in HEADERS):
                current = test_header
                if current not in sections:
                    sections[current] = []
                continue

        if current is not None:
            sections[current].append(raw)
        else:
            # If no header found yet, let's see if we can infer it or if it's a list
            # Some gemma2 outputs are just lists without headers
            pass

    # Heuristic for files without headers: 
    # If we have only 1 list and it ends with "NONE", and no headers found:
    if not sections:
        # Check if text has two distinct lists separated by something or just two parts
        # This is harder to automate reliably without more examples. 
        # For now, let's keep it as is and focus on the ones that DO have headers.
        pass

    return sections

def is_valid_med(med_text: str) -> bool:
    val = med_text.upper()
    BLACKLIST = {
        "UNKNOWN", "NONE", "MEDICATION_CHANGES", "MEDS_CHANGES", "MEDICATIONS",
        "WE HAVE MADE NO CHANGES TO YOUR MEDICATIONS.", "MEDICATION_CHANGES:",
        "DISP #", "REFILLS:", "RX", "SIGNATURE", "DURATION:"
    }
    if not val or val in BLACKLIST:
        return False
    if len(val) < 3: # Too short to be a med
        return False
    if "CHANGES TO YOUR MEDICATIONS" in val:
        return False
    return True

def canonicalise_med_string(text: str) -> str:
    text = clean_line(text).upper().strip()

    # Remove common uncertainty / commentary markers
    text = re.sub(r"\(UNKNOWN IF ON DISCHARGE\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(.*?UNKNOWN.*?\)", "", text, flags=re.IGNORECASE)

    # Remove punctuation that should not drive change detection
    text = re.sub(r"[,:;]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_dose(text: str) -> Optional[str]:
    text = canonicalise_med_string(text)
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(MG|MCG|G|ML|UNITS?|TABS?|CAPS?|PUFFS?|NEB|PATCH)\b", text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def extract_route(text: str) -> Optional[str]:
    text = canonicalise_med_string(text)
    m = re.search(r"\b(PO|IV|IM|SC|SQ|SUBQ|TD|IH|INH|PR|SL|NG|PEG)\b", text)
    if m:
        return m.group(1)
    return None


def extract_frequency(text: str) -> Optional[str]:
    text = canonicalise_med_string(text)

    patterns = [
        r"\bQ\d{1,2}H(?::PRN.*)?\b",
        r"\b(?:DAILY|BID|TID|QID|QHS|QAM|QPM|QOD|WEEKLY|MONTHLY)(?::PRN.*)?\b",
        r"\bONCE DAILY\b",
        r"\bTWICE DAILY\b",
        r"\bTHREE TIMES DAILY\b",
        r"\bFOUR TIMES DAILY\b",
        r"\bEVERY \d+ HOURS?(?::PRN.*)?\b",
        r"\bPRN\b.*",
    ]

    matches = []
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            matches.append(m.group(0).strip())

    return max(matches, key=len) if matches else None

def extract_med_name(med_text: str) -> str:
    """
    Extract a more stable medication name by removing dose/route/frequency tail.
    """
    text = canonicalise_med_string(med_text)

    cut_positions = []

    dose = extract_dose(text)
    route = extract_route(text)
    freq = extract_frequency(text)

    if dose:
        m = re.search(re.escape(dose), text)
        if m:
            cut_positions.append(m.start())

    if route:
        m = re.search(rf"\b{re.escape(route)}\b", text)
        if m:
            cut_positions.append(m.start())

    if freq:
        m = re.search(re.escape(freq), text)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        name = text[:min(cut_positions)].strip()
    else:
        name = text

    # Remove trailing punctuation/spaces
    name = re.sub(r"[,:;]+$", "", name).strip()

    # Optional: remove noisy formulation terms from the name
    noise_words = {
        "TABLET", "TABLETS", "CAPSULE", "CAPSULES", "INHALER", "NEB",
        "NEBULIZER", "NEBULISER", "PATCH", "SYRUP", "SOLUTION",
        "SUSPENSION", "CREAM", "OINTMENT", "GEL", "SPRAY", "DROPS"
    }
    tokens = [t for t in name.split() if t not in noise_words]
    name = " ".join(tokens)

    return re.sub(r"\s+", " ", name).strip()

def compare_same_med(adm_raw: str, dis_raw: str) -> Optional[dict]:
    adm_norm = canonicalise_med_string(adm_raw)
    dis_norm = canonicalise_med_string(dis_raw)

    # If canonically identical, no change
    if adm_norm == dis_norm:
        return None

    adm_dose = extract_dose(adm_raw)
    dis_dose = extract_dose(dis_raw)

    adm_route = extract_route(adm_raw)
    dis_route = extract_route(dis_raw)

    adm_freq = extract_frequency(adm_raw)
    dis_freq = extract_frequency(dis_raw)

    diffs = []
    if adm_dose != dis_dose:
        diffs.append("DOSE_CHANGED")
    if adm_route != dis_route:
        diffs.append("ROUTE_CHANGED")
    if adm_freq != dis_freq:
        diffs.append("FREQUENCY_CHANGED")

    # If parsed fields look same, treat as equivalent formatting noise
    if not diffs:
        return None

    change_type = diffs[0] if len(diffs) == 1 else "MODIFIED"

    return {
        "change_type": change_type,
        "before": adm_raw,
        "after": dis_raw,
        "admission_dose": adm_dose,
        "discharge_dose": dis_dose,
        "admission_route": adm_route,
        "discharge_route": dis_route,
        "admission_frequency": adm_freq,
        "discharge_frequency": dis_freq,
    }

def infer_changes(admission_meds, discharge_meds):
    changes = []

    adm_full = [clean_line(m).upper() for m in admission_meds if is_valid_med(clean_line(m))]
    dis_full = [clean_line(m).upper() for m in discharge_meds if is_valid_med(clean_line(m))]

    blacklist_fragments = ["RX ", "DISP #", "REFILLS:", "DURATION:"]
    adm_full = [m for m in adm_full if not any(b in m for b in blacklist_fragments)]
    dis_full = [m for m in dis_full if not any(b in m for b in blacklist_fragments)]

    # Use dicts keyed by canonical med name
    adm_map = {}
    for m in adm_full:
        key = extract_med_name(m)
        if key and key not in adm_map:
            adm_map[key] = m

    dis_map = {}
    for m in dis_full:
        key = extract_med_name(m)
        if key and key not in dis_map:
            dis_map[key] = m

    adm_names = set(adm_map.keys())
    dis_names = set(dis_map.keys())

    # Present in both -> compare details
    for name in sorted(adm_names & dis_names):
        comparison = compare_same_med(adm_map[name], dis_map[name])
        if comparison:
            changes.append({
                "medication": name,
                **comparison
            })

    # Admission only -> discontinued
    for name in sorted(adm_names - dis_names):
        changes.append({
            "change_type": "DISCONTINUED",
            "medication": name,
            "before": adm_map[name],
            "after": None,
            "admission_dose": extract_dose(adm_map[name]),
            "discharge_dose": None,
            "admission_route": extract_route(adm_map[name]),
            "discharge_route": None,
            "admission_frequency": extract_frequency(adm_map[name]),
            "discharge_frequency": None,
        })

    # Discharge only -> added
    for name in sorted(dis_names - adm_names):
        changes.append({
            "change_type": "ADDED",
            "medication": name,
            "before": None,
            "after": dis_map[name],
            "admission_dose": None,
            "discharge_dose": extract_dose(dis_map[name]),
            "admission_route": None,
            "discharge_route": extract_route(dis_map[name]),
            "admission_frequency": None,
            "discharge_frequency": extract_frequency(dis_map[name]),
        })

    return changes

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

    # Only use input_dir as requested
    all_files = sorted(input_dir.glob("*_meds.txt"))
    print(f"Found {len(all_files)} meds files in {input_dir}")

    if not all_files:
        print("No meds files found.")
        return

    all_inferred_changes = []
    all_med_counts = []
    _blacklist = ["RX ", "DISP #", "REFILLS:", "DURATION:"]

    for path in all_files:
        meta = parse_filename(path)
        if not meta:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")

        # Clean Gemma output if requested
        if args.clean_gemma:
            text = clean_gemma_output(text)

        sections = split_sections(text)

        adm_key = next((k for k in sections if "ADMISSION" in k), None)
        dis_key = next((k for k in sections if "DISCHARGE" in k), None)

        adm_meds = sections.get(adm_key, []) if adm_key else []
        dis_meds = sections.get(dis_key, []) if dis_key else []

        # If no sections found, gemma2 might have returned just a list
        if not sections and text.strip():
            # Attempt to split if there's a clear marker like "NONE" or blank lines between lists
            # But the prompt says "Return ONLY these sections"
            # In our previous check of gemma2 output, it looked like it failed to include headers in some cases?
            # Actually, let's re-examine the gemma2 output.
            pass

        # Count valid medications (same filtering as infer_changes)
        adm_full = [clean_line(m).upper() for m in adm_meds if is_valid_med(clean_line(m))]
        adm_full = [m for m in adm_full if not any(b in m for b in _blacklist)]
        dis_full = [clean_line(m).upper() for m in dis_meds if is_valid_med(clean_line(m))]
        dis_full = [m for m in dis_full if not any(b in m for b in _blacklist)]
        all_med_counts.append({**meta, "adm_med_count": len(adm_full), "dis_med_count": len(dis_full)})

        changes = infer_changes(adm_meds, dis_meds)
        for c in changes:
            all_inferred_changes.append({
                **meta,
                **c
            })

    if all_inferred_changes:
        df = pd.DataFrame(all_inferred_changes)
        output_file = output_dir / "inferred_medication_changes.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} inferred changes to {output_file}")
    else:
        print("\nNo changes inferred.")

    if all_med_counts:
        counts_df = pd.DataFrame(all_med_counts)
        counts_file = output_dir / "medication_counts.csv"
        counts_df.to_csv(counts_file, index=False)
        print(f"Saved medication counts to {counts_file}")

if __name__ == "__main__":
    main()
