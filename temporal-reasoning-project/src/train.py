import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TemporalReasoningModel
from data import TemporalDataset
from tqdm import tqdm

def train():
    model = TemporalReasoningModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    dataset = TemporalDataset('train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.train()
    for epoch in range(3):  # Few epochs for demo
        total_loss = 0
        for batch in tqdm(dataloader):
            text = batch['text'][0]
            label = batch['label'][0]
            optimizer.zero_grad()
            with torch.no_grad():
                model.process_sequence(text)
            outputs = model.answer_query(text)
            loss = criterion(outputs, label.unsqueeze(0))
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), 'models/model.pth')

if __name__ == '__main__':
    train()