import torch
from model import TemporalReasoningModel

def demo():
    model = TemporalReasoningModel()
    # Load trained model if available
    try:
        model.load_state_dict(torch.load('../models/model.pth'))
    except:
        print("No trained model found, using untrained model for demo.")

    # Example document
    document = "John went to the store. He bought milk. Then he returned home."
    print("Processing document:", document)
    model.process_sequence(document)

    # Example query
    query = "What did John buy?"
    print("Query:", query)
    output = model.answer_query(query)
    pred = torch.argmax(output, dim=1).item()
    # Simplified answer mapping
    answers = ["nothing", "milk"]
    print("Predicted answer:", answers[pred] if pred < len(answers) else "unknown")

    # Visualize memory (commented out if matplotlib issues)
    # try:
    #     from utils import visualize_memory
    #     print("Visualizing memory...")
    #     visualize_memory(model.memory)
    # except ImportError:
    #     print("Matplotlib not available, skipping visualization.")

if __name__ == '__main__':
    demo()