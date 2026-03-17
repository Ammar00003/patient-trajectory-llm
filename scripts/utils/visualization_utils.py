from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_OUTPUTS_DIR = PROJECT_ROOT / "llm_outputs"
COHORT_PATH = PROJECT_ROOT / "data" / "selected_cohort.csv"

def get_full_discharge_summary(subject_id: str, hadm_id: str) -> str:
    """Fetch the full discharge summary from the cohort data."""
    try:
        df = pd.read_csv(COHORT_PATH)
        match = df[(df["subject_id"].astype(str) == str(subject_id)) & (df["hadm_id"].astype(str) == str(hadm_id))]
        if not match.empty:
            return str(match["text"].values[0])
    except Exception as e:
        return f"Error loading full DS: {e}"
    return "Full discharge summary not found."

def get_llm_summary(group: str, subject_id: str, hadm_id: str, note_id: str) -> str:
    """Fetch the LLM summarized discharge summary."""
    # The summary is split between meds and timeline files, but let's try to find a general one or combine
    summary = ""
    base_name = f"{group}_subject_{subject_id}_hadm_{hadm_id}_note_{note_id}"

    # Try timeline summary first
    timeline_file = LLM_OUTPUTS_DIR / f"{base_name}_timeline.txt"
    if timeline_file.exists():
        summary += "<b>Timeline Events:</b><br>" + timeline_file.read_text(encoding="utf-8").replace("\n", "<br>")

    # Try meds summary
    meds_file = LLM_OUTPUTS_DIR / f"{base_name}_meds.txt"
    if meds_file.exists():
        if summary: summary += "<br><br>"
        summary += "<b>Medication Changes:</b><br>" + meds_file.read_text(encoding="utf-8").replace("\n", "<br>")

    return summary if summary else "LLM summary not found."

def marker_symbol(record_type: str) -> str:
    """Return a Plotly marker symbol based on record type."""
    if str(record_type).strip().lower() == "medication_change":
        return "square"
    return "circle"

def get_color(record_type: str) -> str:
    """Return a professional color based on record type."""
    if str(record_type).strip().lower() == "medication_change":
        return "#FF6B6B"  # Soft red for medication changes
    return "#4D96FF"      # Clean blue for other events

def build_patient_timeline_figure(patient_df: pd.DataFrame, subject_id: str) -> go.Figure:
    """
    Generate a professional-grade Plotly timeline for a patient with analytics.
    """
    if patient_df.empty:
        return go.Figure()

    # Sort data for chronological order
    patient_df["storetime"] = pd.to_datetime(patient_df["storetime"])
    patient_df = patient_df.sort_values(["storetime", "admission_index", "event_order"]).reset_index(drop=True)

    # Refresh admission indices to be strictly chronological if they aren't
    # Actually, keep the original admission_index if it's already consistent with storetime.
    # Most likely it is.
    admissions = sorted(patient_df["admission_index"].dropna().unique().tolist(), key=lambda x: patient_df[patient_df["admission_index"] == x]["storetime"].min())

    # Calculate days between admissions for analytics
    adm_dates = patient_df[["admission_index", "storetime"]].drop_duplicates().sort_values("storetime")
    adm_dates["prev_storetime"] = adm_dates["storetime"].shift(1)
    adm_dates["days_since_prev"] = (adm_dates["storetime"] - adm_dates["prev_storetime"]).dt.days

    # Setup subplots: Timeline on top, Analytics at bottom if multiple admissions
    if len(admissions) > 1:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.7, 0.3],
            column_widths=[0.7, 0.3],
            specs=[[{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "scatter"}]],
            subplot_titles=("Patient Trajectory Timeline", "Days Between Admissions", "Admission Frequency"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        main_row, main_col = 1, 1
    else:
        fig = go.Figure()
        main_row, main_col = None, None

    # Map admissions to y-axis positions (reversed so first admission is on top)
    y_positions = {adm: len(admissions) - i for i, adm in enumerate(admissions)}

    # Tick text with storetime
    tick_texts = []
    for adm in admissions:
        st = patient_df[patient_df["admission_index"] == adm]["storetime"].iloc[0]
        st_formatted = st.strftime("%b %Y") # e.g. Mar 2018
        tick_texts.append(f"<b>Admission {adm}</b><br>{st_formatted}")

    # Add admission background lines
    for admission_idx in admissions:
        y = y_positions[admission_idx]

        # Draw a faint background line for each admission
        fig.add_trace(
            go.Scatter(
                x=[-0.5, 10.5],
                y=[y, y],
                mode="lines",
                line=dict(color="rgba(200, 200, 200, 0.5)", width=3),
                hoverinfo="skip",
                showlegend=False
            ),
            row=main_row, col=main_col
        )

        adm_df = patient_df[patient_df["admission_index"] == admission_idx].copy()
        n_events = len(adm_df)

        # Distribute events along the x-axis [1, 9]
        if n_events == 1:
            xs = [5]
        else:
            xs = [1 + (8 * i / (n_events - 1)) for i in range(n_events)]

        # Group by record_type to create clear legends
        for r_type in adm_df["record_type"].unique():
            type_df = adm_df[adm_df["record_type"] == r_type]
            type_indices = [i for i, row in enumerate(adm_df.itertuples()) if row.record_type == r_type]
            type_xs = [xs[i] for i in type_indices]

            hover_texts = []
            for _, row in type_df.iterrows():
                hover_texts.append(
                    f"<b>Admission {row['admission_index']}</b><br>"
                    f"<b>Date:</b> {row['storetime'].strftime('%Y-%m-%d')}<br>"
                    f"<b>Type:</b> {row['record_type'].replace('_', ' ').title()}<br>"
                    f"<b>Time:</b> {row['time_phrase']}<br>"
                    f"<b>Event:</b> {row['content']}<br>"
                    f"<extra></extra>"
                )

            fig.add_trace(
                go.Scatter(
                    x=type_xs,
                    y=[y] * len(type_xs),
                    mode="markers+text",
                    name=r_type.replace('_', ' ').title(),
                    marker=dict(
                        size=14,
                        symbol=marker_symbol(r_type),
                        color=get_color(r_type),
                        line=dict(width=1, color="white")
                    ),
                    text=[str(row.event_order) for row in type_df.itertuples()],
                    textposition="top center",
                    textfont=dict(size=10, color="#555"),
                    hovertemplate="%{customdata}",
                    customdata=hover_texts,
                    legendgroup=r_type,
                    showlegend=(admission_idx == admissions[0])  # Show legend only once per type
                ),
                row=main_row, col=main_col
            )

    # --- Analytics: Only if multiple admissions ---
    if len(admissions) > 1:
        valid_diffs = adm_dates.dropna(subset=["days_since_prev"])
        fig.add_trace(
            go.Bar(
                x=[f"Adm {a-1} → {a}" for a in valid_diffs["admission_index"]],
                y=valid_diffs["days_since_prev"],
                marker_color="#4D96FF",
                name="Days Between",
                showlegend=False,
                hovertemplate="Gap: %{y} days<extra></extra>"
            ),
            row=2, col=1
        )

        # --- Analytics: Admission Frequency over time ---
        fig.add_trace(
            go.Scatter(
                x=adm_dates["storetime"],
                y=adm_dates["admission_index"],
                mode="lines+markers",
                line=dict(color="#FF6B6B", width=2),
                marker=dict(size=8),
                name="Adm Index",
                showlegend=False,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Admission #%{y}<extra></extra>"
            ),
            row=2, col=2
        )

    # --- Interaction: Admission Buttons / Details ---
    # We will use multiple updatemenus, one per admission, positioned next to the labels.

    # Calculate y-domain of the main timeline
    # In subplots, row 1 takes [0.3, 1.0] roughly. For single plot, it's [0, 1].
    timeline_y_min, timeline_y_max = (0.2, 1.0) if len(admissions) > 1 else (0.1, 0.95)

    updatemenus = []

    # Static annotations (footer)
    base_annotations = [
        dict(
            x=0.5,
            y=-0.2,  # 👈 move OUTSIDE paper
            xref="paper",
            yref="paper",
            showarrow=False,
            text="Powered by LLM-based clinical note extraction",
            font=dict(size=10.5, color="gray", family="Arial")
        ),
        dict(
            x=0.5,
            y=-0.24,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="Interactive Timeline — Hover over markers for details",
            font=dict(size=11, color="#666")
        )
    ]

    # Global "Clear View" button
    updatemenus.append(dict(
        type="buttons",
        direction="right",
        x=0.01,
        y=1.05,
        showactive=True,
        buttons=[dict(
            label="Clear View",
            method="relayout",
            args=[{"annotations": base_annotations}]
        )],
        bgcolor="white",
        font=dict(size=11)
    ))

    # Calculate button positions based on admission count
    for admission_idx in admissions:
        row_data = patient_df[patient_df["admission_index"] == admission_idx].iloc[0]
        full_ds = get_full_discharge_summary(subject_id, row_data["hadm_id"])
        llm_sum = get_llm_summary(row_data["group"], subject_id, row_data["hadm_id"], row_data["note_id"])

        # Format the summaries
        sum_text = (
            f"<b>ADMISSION {admission_idx} SUMMARY</b><br><br>"
            f"{llm_sum}"
        ).replace("\n", "<br>")

        # Within each admission button set, we will show the full DS without hard truncation if possible
        # or at least a much larger part. Plotly doesn't support scrolling in annotations.
        # However, we can use a very tall annotation if the figure height is also large.

        # Wrap the full DS text to ensure it stays within the box
        import textwrap
        full_ds_wrapped = "<br>".join(["<br>".join(textwrap.wrap(line, width=80)) for line in full_ds.split("\n")])

        ds_text = (
            f"<b>ADMISSION {admission_idx} FULL DS</b><br><br>"
            f"{full_ds_wrapped}"
        ).replace("\n", "<br>")

        y_pos = y_positions[admission_idx]

        # Convert data y-position to paper coordinate roughly
        # Total data range is [0.5, len(admissions) + 0.8]
        # In timeline, we map y_pos to timeline_y_min -> timeline_y_max
        y_range_total = (len(admissions) + 0.8) - 0.5
        norm_y = (y_pos - 0.5) / y_range_total
        paper_y = timeline_y_min + norm_y * (timeline_y_max - timeline_y_min)

        # Create buttons for this admission
        updatemenus.append(dict(
            type="buttons",
            direction="right",
            active=-1,
            x=1.01, # Position it just to the right of the timeline
            y=paper_y,
            xanchor="left",
            yanchor="middle",
            buttons=[
                dict(
                    label="Summary",
                    method="relayout",
                    args=[{"annotations": base_annotations + [
                        dict(
                            text=sum_text, align='left', showarrow=False,
                            xref='paper', yref='paper', x=1.1, y=0.5,
                            xanchor='left', yanchor='middle',
                            bgcolor="rgba(255,255,255,0.95)", bordercolor="#4D96FF",
                            borderwidth=2, width=450, font=dict(size=10)
                        )
                    ]}]
                ),
                dict(
                    label="Full DS",
                    method="relayout",
                    args=[{"annotations": base_annotations + [
                        dict(
                            text=ds_text, align='left', showarrow=False,
                            xref='paper', yref='paper', x=1.1, y=0.5,
                            xanchor='left', yanchor='middle',
                            bgcolor="rgba(255,255,255,0.95)", bordercolor="#888",
                            borderwidth=2, width=450, font=dict(size=10)
                        )
                    ]}]
                )
            ],
            bgcolor="white",
            font=dict(size=9)
        ))

    # Layout enhancements
    layout_update = dict(
        title={
            'text': f"Patient Trajectory Analytics: Subject {subject_id}",
            'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 24, 'family': 'Arial, sans-serif'}
        },
        template="plotly_white",
        height=max(800, 200 * len(admissions)),
        margin=dict(l=100, r=600, t=100, b=140), # Adjusted margins to avoid overlap
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Arial",
            font_color="#333",
            align="left",
            bordercolor="rgba(0, 0, 0, 0.1)"
        ),
        updatemenus=updatemenus,
        annotations=base_annotations
    )

    if len(admissions) > 1:
        # For subplots, we update specifically the main timeline axes (which are xaxis1, yaxis1 by default in make_subplots)
        fig.update_layout(**layout_update)
        fig.update_layout(
            xaxis1=dict(visible=False, range=[-1, 11]),
            yaxis1=dict(
                title="",
                tickmode="array",
                tickvals=[y_positions[a] for a in admissions],
                ticktext=tick_texts,
                range=[0.5, len(admissions) + 0.8],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        )
        # Update axes for analytics subplots
        fig.update_xaxes(title_text="Admission Sequence", row=2, col=1)
        fig.update_yaxes(title_text="Days Gap", row=2, col=1)
        fig.update_xaxes(title_text="Timeline", row=2, col=2)
        fig.update_yaxes(title_text="Admission #", row=2, col=2)
    else:
        # For single plot (go.Figure), we update the main xaxis and yaxis
        fig.update_layout(**layout_update)
        fig.update_layout(
            xaxis=dict(visible=False, range=[-1, 11]),
            yaxis=dict(
                title="",
                tickmode="array",
                tickvals=[y_positions[a] for a in admissions],
                ticktext=tick_texts,
                range=[0.5, len(admissions) + 0.8],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        )

    return fig
