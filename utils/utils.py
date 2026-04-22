import pandas as pd
import plotly.express as px

def plot_timeline(events):
    if not events:
        return None

    df = pd.DataFrame(events)

    if df.empty or 'time' not in df.columns or 'event' not in df.columns:
        return None

    # 🔥 FIX: treat time as numeric (not timestamp)
    df['time'] = df['time'].astype(int)

    # Sort properly
    df = df.sort_values('time')

    # Create simple scatter timeline
    fig = px.scatter(
        df,
        x="time",
        y="event",
        color="subject",
        hover_data=["action"],
        title="Event Timeline"
    )

    # Clean UI
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Event",
        showlegend=True
    )

    return fig