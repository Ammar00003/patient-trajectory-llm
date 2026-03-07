from pathlib import Path
import pandas as pd
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_PATH = PROJECT_ROOT / "data" / "selected_cohort.csv"
OUTPUT_DIR = PROJECT_ROOT / "llm_outputs"

MEDS_PROMPT_PATH = PROJECT_ROOT / "prompts" / "meds_prompt.txt"
TIMELINE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "timeline_prompt.txt"

MODEL_NAME = "mistral:7b"   # change if your Ollama model name differs

# Set to True only for small test reruns
OVERWRITE_EXISTING = False


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, note_text: str) -> str:
    return template.replace("[TEXT]", note_text)


def run_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout.strip()


def save_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    print("Loading cohort...")
    df = pd.read_csv(COHORT_PATH)
    df = df.head(3)
    print("Rows being processed:", len(df))
    meds_prompt_template = load_prompt(MEDS_PROMPT_PATH)
    timeline_prompt_template = load_prompt(TIMELINE_PROMPT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    meds_done = 0
    timeline_done = 0
    skipped = 0
    failed = 0

    # small test
    #df = df.head(3)

    for i, row in df.iterrows():
        subject_id = str(row["subject_id"])
        hadm_id = str(row["hadm_id"])
        note_id = str(row["note_id"])
        group = str(row["group"])
        note_text = str(row["text"])

        base_name = f"{group}_subject_{subject_id}_hadm_{hadm_id}_note_{note_id}"
        meds_file = OUTPUT_DIR / f"{base_name}_meds.txt"
        timeline_file = OUTPUT_DIR / f"{base_name}_timeline.txt"
        error_file = OUTPUT_DIR / f"{base_name}_ERROR.txt"

        both_exist = meds_file.exists() and timeline_file.exists()
        if both_exist and not OVERWRITE_EXISTING:
            skipped += 1
            print(f"[SKIP {i+1}/{total}] {base_name}")
            continue

        try:
            # Prompt A: medications
            meds_prompt = build_prompt(meds_prompt_template, note_text)
            meds_output = run_ollama(meds_prompt)
            save_output(meds_file, meds_output)
            meds_done += 1

            # Prompt B: timeline
            timeline_prompt = build_prompt(timeline_prompt_template, note_text)
            timeline_output = run_ollama(timeline_prompt)
            save_output(timeline_file, timeline_output)
            timeline_done += 1

            print(f"[DONE {i+1}/{total}] {base_name}")

        except Exception as e:
            failed += 1
            save_output(error_file, str(e))
            print(f"[FAIL {i+1}/{total}] {base_name} -> {e}")

    print("\nExtraction complete.")
    print(f"Medication outputs saved: {meds_done}")
    print(f"Timeline outputs saved:   {timeline_done}")
    print(f"Skipped:                  {skipped}")
    print(f"Failed:                   {failed}")


if __name__ == "__main__":
    main()