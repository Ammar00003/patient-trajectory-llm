from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.cleaning_utils import clean_gemma_output

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

def get_llm_summary(group: str, subject_id: str, hadm_id: str, note_id: str, llm_outputs_dir: Path = LLM_OUTPUTS_DIR) -> str:
    """Fetch the LLM summarized discharge summary."""
    # The summary is split between meds and timeline files, but let's try to find a general one or combine
    summary = ""
    base_name = f"{group}_subject_{subject_id}_hadm_{hadm_id}_note_{note_id}"

    # Try timeline summary first
    timeline_file = llm_outputs_dir / f"{base_name}_timeline.txt"
    if timeline_file.exists():
        text = clean_gemma_output(timeline_file.read_text(encoding="utf-8"))
        summary += "<b>Timeline Events:</b><br>" + text.replace("\n", "<br>")

    # Try meds summary
    meds_file = llm_outputs_dir / f"{base_name}_meds.txt"
    if meds_file.exists():
        if summary: summary += "<br><br>"
        text = clean_gemma_output(meds_file.read_text(encoding="utf-8"))
        summary += "<b>Medication Changes:</b><br>" + text.replace("\n", "<br>")

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
            column_widths=[0.6, 0.4], # Increased width for the text area
            specs=[[{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "scatter"}]],
            subplot_titles=("Patient Trajectory Timeline", "Days Between Admissions", "Medications: Admission vs Discharge"),
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
    for i, admission_idx in enumerate(admissions):
        y = y_positions[admission_idx]

        # Alternating background shading for admissions
        if i % 2 == 1:
            fig.add_hrect(
                y0=y - 0.45, y1=y + 0.45,
                fillcolor="#f8f9fa", opacity=1.0, layer="below", line_width=0,
                row=main_row, col=main_col
            )

        # Draw a faint background line for each admission
        fig.add_trace(
            go.Scatter(
                x=[-0.5, 10.5],
                y=[y, y],
                mode="lines",
                line=dict(color="rgba(200, 200, 200, 0.3)", width=2),
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
                # Remove HTML for cleaner plain text display in annotation if we use it
                clean_content = str(row['content']).replace('<br>', ' ').replace('<b>', '').replace('</b>', '')
                time_phrase_line = f"<b>Time:</b> {row['time_phrase']}<br>" if str(row['time_phrase']).strip() else ""
                hover_texts.append(
                    f"<b>Admission {row['admission_index']}</b><br>"
                    f"<b>Date:</b> {row['storetime'].strftime('%Y-%m-%d')}<br>"
                    f"<b>Type:</b> {row['record_type'].replace('_', ' ').title()}<br>"
                    + time_phrase_line +
                    f"<b>Event:</b> {clean_content}<br>"
                    f"<extra></extra>"
                )

            fig.add_trace(
                go.Scatter(
                    x=type_xs,
                    y=[y] * len(type_xs),
                    mode="markers+text",
                    name=r_type.replace('_', ' ').title(),
                    marker=dict(
                        size=15,
                        symbol=marker_symbol(r_type),
                        color=get_color(r_type),
                        line=dict(width=1.5, color="white")
                    ),
                    text=[str(row.event_order) for row in type_df.itertuples()],
                    textposition="top center",
                    textfont=dict(size=10, color="#555"),
                    hovertemplate="%{customdata}",
                    customdata=hover_texts,
                    legendgroup=r_type,
                    showlegend=(admission_idx == admissions[0]),  # Show legend only once per type
                    # Add unique IDs to markers to potentially help with selection
                    ids=[f"marker_{admission_idx}_{row.event_order}" for row in type_df.itertuples()]
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
        fig.add_hline(
            y=30, row=2, col=1,
            line=dict(color="#FFA500", width=1.5, dash="dash"),
            annotation_text="30d", annotation_position="top right",
            annotation_font=dict(color="#FFA500", size=10)
        )
        # Dummy trace so the threshold appears in the legend
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                name="30-day threshold",
                line=dict(color="#FFA500", width=1.5, dash="dash"),
                showlegend=True
            ),
            row=2, col=1
        )

        # --- Medication count line chart ---
        if "adm_med_count" in patient_df.columns and "dis_med_count" in patient_df.columns:
            counts_per_adm = (
                patient_df[["admission_index", "adm_med_count", "dis_med_count"]]
                .drop_duplicates(subset=["admission_index"])
                .sort_values("admission_index")
                .dropna(subset=["adm_med_count", "dis_med_count"])
            )
            if not counts_per_adm.empty:
                x_labels = [f"Adm {a}" for a in counts_per_adm["admission_index"]]
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=counts_per_adm["adm_med_count"],
                        mode="lines+markers",
                        name="On Admission",
                        line=dict(color="#4D96FF", width=2),
                        marker=dict(size=7),
                        legendgroup="med_counts",
                        showlegend=True,
                        hovertemplate="%{x}<br>Meds on admission: %{y}<extra></extra>"
                    ),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=counts_per_adm["dis_med_count"],
                        mode="lines+markers",
                        name="On Discharge",
                        line=dict(color="#FF6B6B", width=2),
                        marker=dict(size=7),
                        legendgroup="med_counts",
                        showlegend=True,
                        hovertemplate="%{x}<br>Meds on discharge: %{y}<extra></extra>"
                    ),
                    row=2, col=2
                )


    # --- Interaction: Admission Buttons / Details ---
    # We will use multiple updatemenus, one per admission, positioned next to the labels.

    # Calculate y-domain of the main timeline
    # In subplots, row 1 takes [0.3, 1.0] roughly. For single plot, it's [0, 1].
    # These are now handled within the loop for better precision
    updatemenus = []

    # Static annotations (footer)
    base_annotations = [
        dict(
            x=0.5,
            y=-0.2,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="Powered by LLM-based clinical note extraction",
            font=dict(size=10.5, color="gray", family="Arial")
        ),
        dict(
            x=0.5,
            y=-0.28,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="Interactive Timeline — Hover over markers for details",
            font=dict(size=11, color="#666")
        )
    ]

    # Calculate button positions based on admission count
    for admission_idx in admissions:
        row_data = patient_df[patient_df["admission_index"] == admission_idx].iloc[0]
        y_pos = y_positions[admission_idx]

        # Use normalized data position within the explicit axis range
        y_range_min = 0.4
        y_range_max = len(admissions) + 0.6
        
        # Normalized position within the axis range (0 to 1)
        norm_y = (y_pos - y_range_min) / (y_range_max - y_range_min)
        
        # Map directly to the paper domain [0.2, 1.0] for subplots or [0.1, 0.95] for single
        # This domain matches where the subplot's y-axis 1 is actually rendered in Plotly's default make_subplots layout
        if len(admissions) > 1:
            # Main timeline in subplots is in top 70% of plot area (row_heights=[0.7, 0.3])
            # Vertical spacing is 0.15. 
            # Subplot row 1 domain is approx [0.35, 1.0]
            paper_y = 0.37 + norm_y * (1.0 - 0.37)
            # Domain for subplot (1, 1) is [0, 1] in x for single, but in subplots it's [0, 1] across both cols 
            # with horizontal_spacing=0.1.
            paper_x_min = 0.0
            paper_x_max = 1.0
        else:
            paper_y = 0.12 + norm_y * (0.93 - 0.12)
            paper_x_min = 0.0
            paper_x_max = 1.0

        adm_df = patient_df[patient_df["admission_index"] == admission_idx].copy()
        n_events = len(adm_df)
        if n_events == 1:
            xs = [5]
        else:
            xs = [1 + (8 * i / (n_events - 1)) for i in range(n_events)]

        # Create Summary/Note buttons for this admission
        updatemenus.append(dict(
            type="buttons",
            direction="right",
            active=-1,
            showactive=False, # Prevent buttons from "phasing" or changing state on click
            x=1.03, # Increased padding from timeline to move buttons further right
            y=paper_y,
            xanchor="left",
            yanchor="middle",
            buttons=[
                dict(
                    label="Summary",
                    method="relayout",
                    args=[{
                        "images[0].source": f"http://localhost:5050/open_summary?subject_id={subject_id}&hadm_id={row_data['hadm_id']}&adm_idx={admission_idx}&group={row_data['group']}&note_id={row_data['note_id']}"
                    }]
                ),
                dict(
                    label="Full Note",
                    method="relayout",
                    args=[{
                        "images[0].source": f"http://localhost:5050/open_ds?subject_id={subject_id}&hadm_id={row_data['hadm_id']}&adm_idx={admission_idx}"
                    }]
                )
            ],
            bgcolor="#ffffff",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(size=10, color="#495057", family="Segoe UI")
        ))

    import copy

    # Build a clean copy of updatemenus (buttons have only their image-source arg, no nested pin state)
    # This is embedded in each button's args[0] so Plotly keeps buttons in position on relayout.
    # NOTE: Plotly.relayout only uses args[0] — args[1] is silently ignored for this method.
    clean_updatemenus = []
    for menu in updatemenus:
        clean_menu = copy.deepcopy(menu)
        for btn in clean_menu.get("buttons", []):
            if "args" in btn:
                btn["args"] = [copy.deepcopy(btn["args"][0])]
        clean_updatemenus.append(clean_menu)

    # Pin state merged into args[0] so every relayout call preserves layout stability
    pin_state = {
        "updatemenus": clean_updatemenus,
        "margin": dict(l=120, r=200, t=110, b=150),
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "hovermode": "closest"
    }

    if len(admissions) > 1:
        pin_state.update({
            "xaxis1": dict(visible=False, range=[-1, 11]),
            "yaxis1": dict(
                tickmode="array",
                tickvals=[y_positions[a] for a in admissions],
                ticktext=tick_texts,
                range=[0.4, len(admissions) + 0.6],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        })
    else:
        pin_state.update({
            "xaxis": dict(visible=False, range=[-1, 11]),
            "yaxis": dict(
                tickmode="array",
                tickvals=[y_positions[a] for a in admissions],
                ticktext=tick_texts,
                range=[0.4, len(admissions) + 0.6],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        })

    # Merge pin_state into every button's args[0] so layout is stable on every click
    for menu in updatemenus:
        for button in menu["buttons"]:
            button["args"][0].update(copy.deepcopy(pin_state))

    # Layout enhancements
    layout_update = dict(
        title={
            'text': f"Patient Trajectory Analytics: <span style='color:#4D96FF'>Subject {subject_id}</span>",
            'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 26, 'family': 'Segoe UI, Arial, sans-serif', 'color': '#333'}
        },
        template="plotly_white",
        height=max(800, 200 * len(admissions)),
        margin=dict(l=120, r=200, t=110, b=150), # Increased right margin from 160
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(size=11, family="Segoe UI")
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Arial",
            font_color="#333",
            align="left",
            bordercolor="rgba(0, 0, 0, 0.1)"
        ),
        images=[dict(
            source="about:blank",
            xref="paper", yref="paper",
            x=0, y=0, sizex=0, sizey=0,
            opacity=0,
            layer="below"
        )],
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
                range=[0.4, len(admissions) + 0.6],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        )
        # Update axes for analytics subplots
        fig.update_xaxes(title_text="Admission Sequence", row=2, col=1)
        fig.update_yaxes(title_text="Days Gap", row=2, col=1)
        fig.update_xaxes(title_text="Admission", row=2, col=2)
        fig.update_yaxes(title_text="Medication Count", row=2, col=2)
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
                range=[0.4, len(admissions) + 0.6],
                gridcolor="rgba(0,0,0,0)",
                zeroline=False
            )
        )

    return fig
