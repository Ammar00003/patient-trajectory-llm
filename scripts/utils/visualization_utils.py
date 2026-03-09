from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

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
    Generate a professional-grade Plotly timeline for a patient.
    
    Args:
        patient_df: DataFrame containing the patient's records.
        subject_id: The ID of the patient.
        
    Returns:
        A Plotly Figure object.
    """
    if patient_df.empty:
        return go.Figure()

    # Sort data for chronological order
    patient_df = patient_df.sort_values(["admission_index", "event_order"]).reset_index(drop=True)
    admissions = sorted(patient_df["admission_index"].dropna().unique().tolist())
    
    fig = go.Figure()

    # Map admissions to y-axis positions (reversed so first admission is on top)
    y_positions = {adm: len(admissions) - i for i, adm in enumerate(admissions)}

    # Add admission background lines
    for admission_idx in admissions:
        y = y_positions[admission_idx]
        
        # Draw a faint background line for each admission
        fig.add_trace(
            go.Scatter(
                x=[-0.5, 10.5],
                y=[y, y],
                mode="lines",
                line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                hoverinfo="skip",
                showlegend=False
            )
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
                )
            )

    # Layout enhancements
    fig.update_layout(
        title={
            'text': f"Patient Trajectory Timeline<br><sup>Subject ID: {subject_id}</sup>",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 24, 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            visible=False,
            range=[-1, 11]
        ),
        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=[y_positions[a] for a in admissions],
            ticktext=[f"<b>Admission {a}</b>" for a in admissions],
            range=[0.5, len(admissions) + 0.8],
            gridcolor="rgba(0,0,0,0)",
            zeroline=False
        ),
        template="plotly_white",
        height=max(500, 180 * len(admissions)),
        margin=dict(l=100, r=50, t=120, b=50),
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
            font_color="#333",  # Dark text for better contrast
            align="left",
            bordercolor="rgba(0, 0, 0, 0.1)"
        ),
        annotations=[
            dict(
                x=0.5, y=-0.12,
                showarrow=False,
                text="Powered by LLM-based clinical note extraction",
                xref="paper", yref="paper",
                font=dict(size=10, color="gray", family="Arial")
            ),
            dict(
                x=0.5, y=-0.06,
                showarrow=False,
                text="Interactive Timeline — Hover over markers for details",
                xref="paper", yref="paper",
                font=dict(size=11, color="#666")
            )
        ]
    )

    return fig
