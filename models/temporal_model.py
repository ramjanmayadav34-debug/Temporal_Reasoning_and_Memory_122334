from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import numpy as np

class TemporalModel:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression(max_iter=200, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.subject_sequences: Dict[str, List[str]] = defaultdict(list)
        self.global_transitions: Dict[str, Counter] = defaultdict(Counter)
        self.action_vocab: Dict[str, int] = {}

    def add_event(self, event_text: str, time: int, subject: str, action: str):
        event = {
            "event": event_text,
            "time": time,
            "subject": subject,
            "action": action
        }
        self.events.append(event)
        self.subject_sequences[subject].append(action)
        if len(self.subject_sequences[subject]) > 1:
            prev = self.subject_sequences[subject][-2]
            self.global_transitions[prev][action] += 1
        self.trained = False  # Mark for retraining

    def _build_features(self, events_list: List[Dict[str, Any]]) -> np.ndarray:
        features = []
        for event in events_list:
            text = event.get("event", "")
            action = event.get("action", "")
            time = event.get("time", 0)
            subject = event.get("subject", "")
            
            # Combine text features
            combined_text = f"{text} {action} {subject}"
            text_vec = self.vectorizer.transform([combined_text]).toarray()[0]
            
            # Time feature (normalized)
            time_norm = (time - 2000) / 50  # Rough normalization
            
            # Sequence feature: last action index
            last_action = self.subject_sequences.get(subject, [])[-1] if self.subject_sequences.get(subject) else ""
            action_idx = self.action_vocab.get(last_action, 0)
            
            feature = np.concatenate([text_vec, [time_norm, action_idx]])
            features.append(feature)
        
        return np.array(features)

    def train_model(self):
        if len(self.events) < 3:
            return

        events = sorted(self.events, key=lambda x: x["time"])

        # Build action vocab
        all_actions = [e["action"] for e in events]
        self.action_vocab = {action: i for i, action in enumerate(set(all_actions))}

        # Fit vectorizer first
        texts = [f"{e['event']} {e['action']} {e['subject']}" for e in events]
        self.vectorizer.fit(texts)

        X = []
        y = []

        for i in range(len(events) - 1):
            # Use sequence of last 3 events for context
            start = max(0, i-2)
            seq_events = events[start:i+1]
            feature = self._build_features(seq_events)[-1]  # Last event's feature
            X.append(feature)
            y.append(events[i + 1]["action"])

        if len(set(y)) < 2:
            return

        X = np.array(X)
        X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        self.trained = True

    def _markov_predict(self, subject: str, last_action: str) -> List[Tuple[str, float]]:
        if subject in self.subject_sequences and len(self.subject_sequences[subject]) > 1:
            # Subject-specific transitions
            transitions = defaultdict(Counter)
            seq = self.subject_sequences[subject]
            for i in range(len(seq)-1):
                transitions[seq[i]][seq[i+1]] += 1
            if last_action in transitions:
                total = sum(transitions[last_action].values())
                probs = [(action, count/total) for action, count in transitions[last_action].items()]
                return sorted(probs, key=lambda x: x[1], reverse=True)[:3]
        
        # Fallback to global
        if last_action in self.global_transitions:
            total = sum(self.global_transitions[last_action].values())
            probs = [(action, count/total) for action, count in self.global_transitions[last_action].items()]
            return sorted(probs, key=lambda x: x[1], reverse=True)[:3]
        
        return []

    def predict_next_event(self, events_list: List[Dict[str, Any]]) -> str:
        if not events_list:
            return "No data available."

        # Retrain if needed
        if not self.trained:
            self.train_model()

        events = sorted(events_list, key=lambda x: x.get("time", 0))
        last_event = events[-1]

        text = last_event.get("event", "")
        subject = last_event.get("subject", "Someone")
        year = last_event.get("time", "future")
        last_action = last_event.get("action", "")

        predictions = []

        # Markov prediction
        markov_preds = self._markov_predict(subject, last_action)
        if markov_preds:
            for action, prob in markov_preds[:3]:
                predictions.append((action, prob, "sequence pattern"))

        # ML prediction
        if self.trained and len(events) > 1:
            seq_events = events[-3:] if len(events) >= 3 else events
            feature = self._build_features(seq_events)[-1]
            feature = self.scaler.transform([feature])
            probs = self.model.predict_proba(feature)[0]
            classes = self.model.classes_
            
            top_indices = np.argsort(probs)[-3:][::-1]
            for idx in top_indices:
                action = classes[idx]
                conf = probs[idx]
                predictions.append((action, conf, "ML model"))

        if not predictions:
            return f"After {year}, not enough data to predict."

        # Combine and rank
        action_scores = defaultdict(list)
        for action, score, source in predictions:
            action_scores[action].append((score, source))

        # Average scores or take max
        ranked = []
        for action, scores_sources in action_scores.items():
            avg_score = np.mean([s[0] for s in scores_sources])
            sources = [s[1] for s in scores_sources]
            ranked.append((action, avg_score, sources))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Generate explanation
        top3 = ranked[:3]
        explanation = f"Based on {subject}'s history and global patterns, the most likely next events after {year} are:\n"
        for i, (action, conf, sources) in enumerate(top3, 1):
            conf_pct = round(conf * 100, 1)
            source_str = ", ".join(set(sources))
            explanation += f"{i}. '{action}' (confidence: {conf_pct}%, based on {source_str})\n"

        return explanation.strip()