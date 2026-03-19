# Temporal Reasoning and Memory Formation in Neural Sequence Models

This project explores improving temporal reasoning and long-range memory in sequence models like Transformers by integrating external memory modules and event segmentation.

## Project Structure

- `src/`: Source code
  - `model.py`: Model implementations (Transformer, Memory Bank, etc.)
  - `data.py`: Data loading and preprocessing
  - `train.py`: Training script
  - `evaluate.py`: Evaluation script
  - `utils.py`: Utility functions
- `data/`: Datasets
  - `raw/`: Raw data
  - `processed/`: Processed data
- `models/`: Saved model checkpoints
- `notebooks/`: Jupyter notebooks for exploration
- `docs/`: Documentation and report
- `requirements.txt`: Python dependencies

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: Run preprocessing scripts in `src/data.py`
3. Train model: `python src/train.py`
4. Evaluate: `python src/evaluate.py`

## Demo

Run the end-to-end demo: `python src/demo.py`