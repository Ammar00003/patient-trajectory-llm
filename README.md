# Patient Trajectory Extraction using LLMs

This project extracts structured clinical information from MIMIC-IV discharge summaries using a locally hosted large language model (Mistral 7B).

The extracted information is used to construct patient trajectory timelines across multiple hospital admissions.

## Project Goals

- Automatically extract clinical events from discharge summaries
- Identify medication changes and transitional issues
- Construct patient trajectories across admissions
- Visualise trajectories as timelines

## Dataset

This project uses the MIMIC-IV clinical database from PhysioNet.

⚠️ Patient data is not included in this repository due to data use agreements.

## Model

Local LLM:
- Mistral 7B
- Running via Ollama

## Pipeline

1. Select patient cohort (25 single-admission, 25 multi-admission)
2. Run LLM extraction on discharge summaries
3. Parse extracted outputs
4. Construct patient trajectories
5. Visualise timelines

## Repository Structure
prompts/        LLM extraction prompts
scripts/        data processing scripts
data/           dataset location (not tracked by git)
raw_notes/      extracted discharge summaries
llm_outputs/    model outputs
notebooks/      analysis and visualisation
docs/           report notes
