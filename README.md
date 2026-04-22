# Temporal Reasoning System

An intelligent system for analyzing temporal event sequences, predicting future events, and querying historical data.

## Features

- **Advanced NLP Processing**: Extracts subjects and actions from unstructured event text, normalizes similar actions.
- **Sequence Learning**: Uses Markov chains for subject-specific and global event transitions.
- **Context-Aware Prediction**: Combines text features, action sequences, and temporal information.
- **Top-3 Predictions**: Provides multiple predictions with confidence scores and human-like explanations.
- **Robust Data Handling**: Auto-detects event and time columns in CSV/JSON, handles missing fields.
- **Incremental Learning**: Model updates automatically as new events are added.
- **Fast Training**: Optimized for quick retraining with minimal dependencies.
- **Streamlit UI**: Interactive web interface for data upload, querying, and visualization.

## Improvements Made

1. **Enhanced NLP**:
   - Regex-based action extraction for verbs like join, promote, leave, etc.
   - Action normalization (e.g., "promoted" → "promote", "started" → "join").

2. **Sequence Modeling**:
   - Markov chains for predicting next actions based on personal history.
   - Global transition patterns for subjects with limited history.

3. **Feature Engineering**:
   - Combined text + action + subject + time features.
   - TF-IDF vectorization on enriched text.
   - Sequence context (last 3 events).

4. **Prediction Output**:
   - Top 3 predictions with confidence percentages.
   - Explanations based on sequence patterns or ML model.

5. **Robustness**:
   - Auto-detection of columns in CSV files.
   - Handles missing event/time fields gracefully.
   - Works with any JSON/CSV dataset structure.

6. **Performance**:
   - Minimal dependencies: streamlit, pandas, plotly, scikit-learn, numpy.
   - Fast training suitable for Streamlit deployment.
   - Incremental updates without full retraining.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Upload your event data (JSON or CSV), add events manually, query the system, and get predictions.

## Architecture

- `data/data_handler.py`: Data loading, NLP extraction, normalization.
- `models/temporal_model.py`: Sequence modeling, ML prediction.
- `memory/memory_module.py`: Event storage and querying.
- `utils/utils.py`: Visualization utilities.
- `app.py`: Streamlit interface.