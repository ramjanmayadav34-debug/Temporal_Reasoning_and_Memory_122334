from data.data_handler import DataHandler
from models.temporal_model import TemporalModel

# Test the upgraded system
dh = DataHandler()
model = TemporalModel()

# Add sample events
events_data = [
    ("John joined the company", 2020, "John", "join"),
    ("John was promoted to manager", 2022, "John", "promote"),
    ("John attended a conference", 2023, "John", "attend"),
    ("Alice started working", 2021, "Alice", "join"),
    ("Alice got a raise", 2023, "Alice", "receive"),
    ("Bob became team lead", 2024, "Bob", "become"),
]

for text, time, subj, act in events_data:
    dh.add_event(text, time)
    model.add_event(text, time, subj, act)

# Test prediction
events_list = [e.to_dict() for e in dh.get_events()]
prediction = model.predict_next_event(events_list)
print("Prediction:")
print(prediction)