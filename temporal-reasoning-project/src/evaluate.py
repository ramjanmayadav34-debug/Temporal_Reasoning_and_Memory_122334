import torch
from model import TemporalReasoningModel
from data import TemporalDataset
from torch.utils.data import DataLoader

def evaluate():
    model = TemporalReasoningModel()
    try:
        model.load_state_dict(torch.load('../models/model.pth'))
        print("Loaded trained model.")
    except FileNotFoundError:
        print("No trained model found, evaluating untrained model.")
    model.eval()

    dataset = TemporalDataset('validation')
    dataloader = DataLoader(dataset, batch_size=1)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            text = batch['text'][0]
            label = batch['label'][0]
            model.process_sequence(text)
            outputs = model.answer_query(text)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
            total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    # Additional metrics: Temporal Ordering Accuracy (placeholder)
    # Memory Recall: Check if relevant memories are retrieved

if __name__ == '__main__':
    evaluate()