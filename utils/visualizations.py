"""
Interactive visualization suite built with Plotly.
Modern styling, accessible color schemes, and responsive layout configurations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Theme palette
PALETTE = ["#6366F1", "#10B981", "#F59E0B", "#EC4899", "#3B82F6", "#8B5CF6", "#14B8A6", "#F43F5E"]
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)"),
    legend=dict(font=dict(color="#cbd5e1")),
    title=dict(font=dict(size=16, color="#f8fafc"))
)

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, orientation: str = "v", color: str = None) -> go.Figure:
    """Generates a stylish bar chart."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        color=color,
        color_discrete_sequence=PALETTE,
        text_auto=True
    )
    fig.update_layout(**DARK_LAYOUT)
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.2)")
    return fig

def plot_pie(df: pd.DataFrame, names: str, values: str, title: str, hole: float = 0.45) -> go.Figure:
    """Generates a modern donut / pie chart."""
    fig = px.pie(
        df,
        names=names,
        values=values,
        title=title,
        hole=hole,
        color_discrete_sequence=PALETTE
    )
    fig.update_layout(**DARK_LAYOUT)
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
    return fig

def plot_histogram(df: pd.DataFrame, x: str, title: str, nbins: int = 25, color: str = None) -> go.Figure:
    """Generates a histogram with marginal boxplot."""
    fig = px.histogram(
        df,
        x=x,
        title=title,
        nbins=nbins,
        color=color,
        marginal="box",
        color_discrete_sequence=PALETTE,
        opacity=0.85
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def plot_box(df: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
    """Generates an aesthetic box plot."""
    fig = px.box(
        df,
        x=x,
        y=y,
        title=title,
        color=color or x,
        color_discrete_sequence=PALETTE,
        points="outliers"
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], title: str = "Correlation Heatmap (Pearson r)") -> go.Figure:
    """Generates an annotated correlation heatmap."""
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def plot_time_series(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Generates a smooth line chart with markers."""
    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        markers=True,
        color_discrete_sequence=["#6366F1"]
    )
    fig.update_layout(**DARK_LAYOUT)
    fig.update_traces(line=dict(width=3))
    return fig

def plot_regression_actual_vs_pred(y_true, y_pred, title: str = "Actual vs. Predicted Demand") -> go.Figure:
    """Plots actual vs predicted values with reference diagonal line."""
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#6366F1', size=8, opacity=0.7, line=dict(width=1, color='#ffffff'))
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Fit Line (y = x)',
        line=dict(color='#10B981', dash='dash', width=2)
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    return fig

def plot_regression_residuals(y_true, y_pred, title: str = "Residual Distribution (y - ŷ)") -> go.Figure:
    """Plots residual distribution for regression diagnostics."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig = px.histogram(
        x=residuals,
        nbins=25,
        title=title,
        color_discrete_sequence=["#EC4899"],
        marginal="box"
    )
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Residuals (Error)",
        yaxis_title="Frequency"
    )
    return fig

def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = "Confusion Matrix") -> go.Figure:
    """Plots an annotated confusion matrix heatmap."""
    fig = px.imshow(
        cm_df.values,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=cm_df.columns,
        y=cm_df.index,
        text_auto=True,
        title=title,
        color_continuous_scale="Purples"
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def plot_feature_importance(features: list, importances: list, title: str = "Feature Importance") -> go.Figure:
    """Bar chart for model feature importance or coefficients."""
    df = pd.DataFrame({"Feature": features, "Importance": importances})
    df = df.sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        color="Importance",
        color_continuous_scale="Viridis",
        text_auto=".3f"
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig

def plot_pca_2d(pca_df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str = "PCA 2D Projection") -> go.Figure:
    """Plots 2D PCA scatter plot with categorical color grouping."""
    fig = px.scatter(
        pca_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_sequence=PALETTE,
        hover_data=pca_df.columns
    )
    fig.update_layout(**DARK_LAYOUT)
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color='#ffffff')))
    return fig

def plot_cluster_scatter(df: pd.DataFrame, x: str, y: str, cluster_col: str, title: str = "Cluster Distribution") -> go.Figure:
    """Plots cluster groups."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=cluster_col,
        title=title,
        color_discrete_sequence=PALETTE,
        hover_data=df.columns
    )
    fig.update_layout(**DARK_LAYOUT)
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    return fig
