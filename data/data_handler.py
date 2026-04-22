import json
import pandas as pd
import re
from typing import List, Dict, Any

class Event:
    def __init__(self, event_text: str, time: int, subject: str = None, action: str = None):
        self.event_text = event_text
        self.time = time
        self.subject = subject or self.extract_subject()
        self.action = action or self.extract_action()

    def extract_subject(self) -> str:
        # Improved: Look for names (capitalized words), or pronouns
        words = self.event_text.split()
        for word in words:
            if word[0].isupper() and len(word) > 1:
                return word.rstrip('.,')
        # Fallback to first word
        return words[0].rstrip('.,') if words else "unknown"

    def extract_action(self) -> str:
        # Improved: Use regex to find verbs/actions
        action_patterns = [
            r'\b(joined|started|left|quit|resigned|promoted|demoted|attended|got|received|became|was|were)\b',
            r'\b(hired|fired|laid off|retired|transferred|moved|changed)\b'
        ]
        for pattern in action_patterns:
            match = re.search(pattern, self.event_text.lower())
            if match:
                return match.group(1)
        # Fallback
        return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event': self.event_text,
            'time': self.time,
            'subject': self.subject,
            'action': self.action
        }

class DataHandler:
    def __init__(self):
        self.events: List[Event] = []
        self.action_normalizer = {
            'joined': 'join',
            'started': 'join',
            'hired': 'join',
            'left': 'leave',
            'quit': 'leave',
            'resigned': 'leave',
            'fired': 'leave',
            'laid off': 'leave',
            'retired': 'leave',
            'promoted': 'promote',
            'demoted': 'demote',
            'got': 'receive',
            'received': 'receive',
            'became': 'become',
            'was': 'become',
            'were': 'become',
            'attended': 'attend',
            'transferred': 'transfer',
            'moved': 'transfer',
            'changed': 'change'
        }

    def normalize_action(self, action: str) -> str:
        return self.action_normalizer.get(action.lower(), action.lower())

    def load_json(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        for item in data:
            event_text = item.get('event', item.get('text', ''))
            time = item.get('time', item.get('timestamp', 0))
            if event_text and time:
                event = Event(event_text, time)
                event.action = self.normalize_action(event.action)
                self.events.append(event)

    def load_csv(self, file_path: str):
        df = pd.read_csv(file_path)
        # Auto-detect columns
        event_col = None
        time_col = None
        for col in df.columns:
            if 'event' in col.lower() or 'text' in col.lower():
                event_col = col
            elif 'time' in col.lower() or 'year' in col.lower() or 'date' in col.lower():
                time_col = col
        if not event_col or not time_col:
            raise ValueError("Could not auto-detect event and time columns in CSV.")
        
        for _, row in df.iterrows():
            event_text = str(row[event_col])
            time_str = str(row[time_col])
            # Extract year if date
            match = re.search(r'\d{4}', time_str)
            time = int(match.group()) if match else int(time_str)
            event = Event(event_text, time)
            event.action = self.normalize_action(event.action)
            self.events.append(event)

    def add_event(self, event_text: str, time: int) -> Event:
        event = Event(event_text, time)
        event.action = self.normalize_action(event.action)
        self.events.append(event)
        return event

    def get_events(self) -> List[Event]:
        return self.events

    def sort_by_time(self):
        self.events.sort(key=lambda x: x.time)

    def get_events_after(self, time: int) -> List[Event]:
        return [e for e in self.events if e.time > time]

    def get_events_before(self, time: int) -> List[Event]:
        return [e for e in self.events if e.time < time]

    def get_events_by_subject(self, subject: str) -> List[Event]:
        return [e for e in self.events if e.subject.lower() == subject.lower()]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.events])

    def get_subjects(self) -> List[str]:
        return list(set(e.subject for e in self.events))

    def get_actions(self) -> List[str]:
        return list(set(e.action for e in self.events))