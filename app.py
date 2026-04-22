import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO

from data.data_handler import DataHandler
from models.temporal_model import TemporalModel
from memory.memory_module import MemoryModule, TemporalReasoner
from utils.utils import plot_timeline


# ==============================
#  PAGE CONFIG
# ==============================
st.set_page_config(page_title="Temporal Reasoning System", layout="wide")


# ==============================
#  SESSION STATE INIT
# ==============================
if "data_handler" not in st.session_state:
    st.session_state.data_handler = DataHandler()

if "model" not in st.session_state:
    st.session_state.model = TemporalModel()

if "memory" not in st.session_state:
    st.session_state.memory = MemoryModule()

if "reasoner" not in st.session_state:
    st.session_state.reasoner = TemporalReasoner(st.session_state.memory)


# ==============================
#  UNIVERSAL NORMALIZER
# ==============================
def normalize_json(data):
    normalized = []
    warnings = []

    def extract(obj):
        event_keys = ["event", "text", "message", "description", "title"]
        time_keys = ["time", "timestamp", "year", "date", "created_at"]

        event = None
        time = None

        for k in event_keys:
            if k in obj and obj[k]:
                event = str(obj[k])
                break

        for k in time_keys:
            if k in obj and obj[k]:
                val = str(obj[k])
                match = re.search(r"\d{4}", val)
                if match:
                    time = int(match.group())
                break

        if event and time:
            return {"event": event, "time": time}
        return None

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                res = extract(item)
                if res:
                    normalized.append(res)
                else:
                    warnings.append(f"Event {i+1}: Missing event/time")
    elif isinstance(data, dict):
        for key in data:
            if isinstance(data[key], dict):
                res = extract(data[key])
                if res:
                    normalized.append(res)

    return normalized, warnings


# ==============================
# SIDEBAR UI
# ==============================
st.sidebar.title("📂 Data Input")

event_input = st.sidebar.text_input("Add Event (event | time)")

if st.sidebar.button("Add Event"):
    try:
        text, time = event_input.split("|")
        time = int(time.strip())

        obj = st.session_state.data_handler.add_event(text.strip(), time)
        st.session_state.model.add_event(obj.event_text, obj.time, obj.subject, obj.action)
        st.session_state.memory.add_event(obj.to_dict())

        st.sidebar.success("Event added ✅")
    except:
        st.sidebar.error("Use format: event | time")


# ==============================
#   FILE UPLOAD
# ==============================
file = st.sidebar.file_uploader("Upload JSON / CSV", type=["json", "csv"])

if file:
    try:
        # Save temp file
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        
        if file.name.endswith(".csv"):
            st.session_state.data_handler.load_csv(temp_path)
        else:
            st.session_state.data_handler.load_json(temp_path)
        
        # Add to model and memory
        for event in st.session_state.data_handler.events[-len(st.session_state.data_handler.events):]:  # All loaded
            st.session_state.model.add_event(event.event_text, event.time, event.subject, event.action)
            st.session_state.memory.add_event(event.to_dict())
        
        st.sidebar.success(f"{len(st.session_state.data_handler.events)} events loaded 🚀")
        
        # Clean up
        import os
        os.remove(temp_path)
        
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


# ==============================
# SAMPLE DATA BUTTON
# ==============================
if st.sidebar.button("Load Sample Data"):
    sample = [
        {"event": "John joined the company", "time": 2020},
        {"event": "Alice started working", "time": 2021},
        {"event": "John was promoted to manager", "time": 2022},
        {"event": "Alice got a raise", "time": 2023},
        {"event": "Bob became team lead", "time": 2024},
        {"event": "John attended a conference", "time": 2023},
        {"event": "Alice transferred to new department", "time": 2024},
        {"event": "Bob left the company", "time": 2025},
    ]

    for e in sample:
        obj = st.session_state.data_handler.add_event(e["event"], e["time"])
        st.session_state.model.add_event(obj.event_text, obj.time, obj.subject, obj.action)
        st.session_state.memory.add_event(obj.to_dict())

    st.sidebar.success("Sample data loaded ✅")


# ==============================
#  MAIN UI
# ==============================
st.title(" Temporal Reasoning and Memory System")

# QUERY
st.subheader("🔍 Query System")
query = st.text_input("Ask anything about events")

if st.button("Ask"):
    answer = st.session_state.reasoner.answer_query(query)
    st.success(answer)


# EVENTS TABLE
st.subheader("📊 Stored Events")

events = st.session_state.data_handler.get_events()

if events:
    df = pd.DataFrame([e.to_dict() for e in events])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No events yet")


# TIMELINE
st.subheader("📈 Timeline")

fig = plot_timeline([e.to_dict() for e in events])
if fig:
    st.plotly_chart(fig, use_container_width=True)


# MEMORY
st.subheader("🧠 Memory Status")
st.write(f"Total events: {len(events)}")


# PREDICTION
st.subheader("🔮 Prediction")

if st.button("Predict Next Event"):
    prediction = st.session_state.model.predict_next_event(
        [e.to_dict() for e in events]
    )
    st.markdown(prediction)