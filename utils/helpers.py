"""
Helper utilities for UI styling, formatting, and data export.
"""

import streamlit as st
import pandas as pd
import io

def inject_custom_css():
    """Injects modern, polished CSS styles for the Streamlit dashboard."""
    st.markdown(
        """
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Metric / KPI Card Styling */
        .kpi-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .kpi-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            flex: 1 1 calc(25% - 1rem);
            min-width: 200px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .kpi-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.2rem;
        }

        .kpi-subtitle {
            font-size: 0.78rem;
            color: #64748b;
        }

        /* Status Badge Pills */
        .badge-pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-align: center;
        }

        .badge-success {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-warning {
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .badge-danger {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .badge-info {
            background-color: rgba(59, 130, 246, 0.15);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        /* Lab Banner */
        .lab-banner {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
            border: 1px solid rgba(129, 140, 248, 0.3);
            border-radius: 12px;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.75rem;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.15);
        }

        .lab-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        /* Section Header */
        .section-header {
            border-left: 4px solid #6366f1;
            padding-left: 0.75rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        /* Formula Card */
        .formula-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_kpi_card(title: str, value: str | int | float, subtitle: str = "", border_color: str = "#6366f1"):
    """HTML renderer for high-aesthetic KPI cards."""
    return f"""
    <div style="background: rgba(30, 41, 59, 0.7); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); border-top: 3px solid {border_color}; padding: 16px 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
        <div style="color: #94a3b8; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        <div style="color: #f8fafc; font-size: 28px; font-weight: 800; margin: 4px 0;">{value}</div>
        <div style="color: #64748b; font-size: 12px;">{subtitle}</div>
    </div>
    """

def render_lab_banner(experiment_num: int, title: str, description: str, tags: list[str] = None):
    """Renders academic experiment header banner."""
    tag_html = ""
    if tags:
        tag_html = " ".join([f"<span style='background: rgba(255,255,255,0.15); padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 6px;'>{t}</span>" for t in tags])
    
    st.markdown(
        f"""
        <div class="lab-banner">
            <div class="lab-badge">Practical Experiment {experiment_num}</div>
            <h2 style="margin: 4px 0 8px 0; font-size: 22px; font-weight: 800; color: #ffffff;">{title}</h2>
            <p style="margin: 0 0 10px 0; font-size: 14px; color: #e0e7ff; line-height: 1.5;">{description}</p>
            <div>{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def download_csv_button(df: pd.DataFrame, filename: str, button_label: str = "📥 Download CSV Report"):
    """Renders a download button for any DataFrame."""
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=button_label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )
