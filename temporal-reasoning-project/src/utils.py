import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_memory(memory_bank):
    if memory_bank.pointer == 0:
        print("No memories to visualize")
        return
    embeddings = memory_bank.memory[:memory_bank.pointer, :768].detach().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=range(len(reduced)), cmap='viridis')
    plt.colorbar(label='Memory Index')
    plt.title('Memory Embeddings Visualization')
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.show()

def plot_event_timeline(times):
    plt.figure(figsize=(10, 2))
    plt.eventplot(times, orientation='horizontal', colors='blue')
    plt.title('Event Timeline')
    plt.xlabel('Time')
    plt.show()

# Example usage in demo
# from model import TemporalReasoningModel
# model = TemporalReasoningModel()
# model.process_sequence("Event 1. Event 2.")
# visualize_memory(model.memory)