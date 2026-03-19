import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.pooler_output  # [batch_size, hidden_size]

class EventSegmentation:
    @staticmethod
    def segment(text):
        # Simple sentence-based segmentation
        sentences = text.split('.')
        return [s.strip() for s in sentences if s.strip()]

class ExternalMemoryModule(nn.Module):
    def __init__(self, mem_size=100, embed_dim=768):
        super().__init__()
        self.mem_size = mem_size
        self.embed_dim = embed_dim
        self.memory = torch.zeros(mem_size, embed_dim * 2)  # key + value
        self.pointer = 0
        self.times = torch.zeros(mem_size)  # timestamps

    def write(self, key, value, time):
        idx = self.pointer
        self.memory[idx] = torch.cat([key, value], dim=-1)
        self.times[idx] = time
        self.pointer = (self.pointer + 1) % self.mem_size

    def read(self, query):
        if self.pointer == 0:
            return torch.zeros(self.embed_dim)
        # Cosine similarity for retrieval
        keys = self.memory[:self.pointer, :self.embed_dim]
        values = self.memory[:self.pointer, self.embed_dim:]
        similarities = torch.cosine_similarity(query.unsqueeze(0), keys, dim=-1)
        idx = torch.argmax(similarities)
        return values[idx]

class MemoryConsolidation:
    @staticmethod
    def consolidate(memory_module, threshold=10):
        # Simple consolidation: average embeddings if too many
        if memory_module.pointer > threshold:
            # Average recent memories
            recent = memory_module.memory[:memory_module.pointer]
            avg = torch.mean(recent, dim=0)
            memory_module.memory[0] = avg
            memory_module.pointer = 1

class TemporalReasoningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.memory = ExternalMemoryModule()
        self.segmenter = EventSegmentation()
        self.consolidator = MemoryConsolidation()
        self.classifier = nn.Linear(768 * 2, 2)  # For binary tasks

    def process_sequence(self, sequence, time=0):
        events = self.segmenter.segment(sequence)
        for event in events:
            embedding = self.encoder(event).squeeze(0)  # [hidden_size]
            self.memory.write(embedding, embedding, time)
            time += 1
        self.consolidator.consolidate(self.memory)

    def answer_query(self, query):
        query_emb = self.encoder(query).squeeze(0)  # [hidden_size]
        memory_out = self.memory.read(query_emb)  # [hidden_size]
        combined = torch.cat([query_emb, memory_out], dim=-1)  # [2*hidden_size]
        return self.classifier(combined.unsqueeze(0))  # [1, 2]