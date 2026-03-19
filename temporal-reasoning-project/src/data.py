from datasets import load_dataset
import torch
from torch.utils.data import Dataset

class TemporalDataset(Dataset):
    def __init__(self, split='train', max_samples=100):
        # Load a simple temporal dataset, e.g., MRPC for demo
        dataset = load_dataset('glue', 'mrpc', split=split)
        self.dataset = dataset.select(range(min(max_samples, len(dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item['sentence1'] + ' ' + item['sentence2'],
            'label': item['label']
        }

def preprocess_data():
    # For custom datasets, add preprocessing here
    # E.g., segment events, tokenize
    pass

# For synthetic log data
def generate_synthetic_logs(num_logs=100):
    logs = []
    for i in range(num_logs):
        log = f"Event {i}: Server started. User logged in. Database updated."
        logs.append(log)
    return logs