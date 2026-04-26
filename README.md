# Patient Trajectory Extraction using LLMs

This project extracts structured clinical information from MIMIC-IV discharge summaries using locally hosted large language models (Gemma 4, Mistral 7B).

The extracted information is used to construct patient trajectory timelines across multiple hospital admissions, visualised through an interactive GUI viewer with analytics charts.

## Project Goals

- Automatically extract clinical events from discharge summaries
- Identify medication changes and transitional issues
- Construct patient trajectories across admissions
- Visualise trajectories as interactive timelines with admission analytics

## Dataset

This project uses the MIMIC-IV clinical database from PhysioNet.

⚠️ Patient data is not included in this repository due to data use agreements.

## Models

Local LLMs running via Ollama:
- Gemma 4 (`gemma4:e2b-it-q4_K_M`) — primary
- Mistral 7B — secondary / comparison

## Pipeline

1. Select patient cohort (25 single-admission, 25 multi-admission)
2. Extract relevant sections from discharge notes
3. Run LLM extraction (Gemma 4 or Mistral 7B)
4. Parse and clean extracted outputs
5. Infer medication changes and counts per admission
6. Build patient trajectory CSV
7. Launch interactive timeline viewer

## Repository Structure

```
prompts/              LLM extraction prompts
scripts/              data processing and extraction scripts
scripts/utils/        shared cleaning and visualisation utilities
data/                 dataset location (not tracked)
raw_notes/            extracted discharge summaries
llm_outputs/          Mistral 7B outputs
llm_outputs_gemma4/   Gemma 4 outputs
notebooks/            analysis notebooks
docs/                 report notes
```
