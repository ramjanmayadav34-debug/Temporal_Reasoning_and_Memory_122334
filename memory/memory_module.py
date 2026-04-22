from typing import List, Dict, Any
import re


class MemoryModule:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add_event(self, event: Dict[str, Any]):
        self.events.append(event)


class TemporalReasoner:
    def __init__(self, memory: MemoryModule):
        self.memory = memory

    # 🔥 Helper: format results nicely
    def format_results(self, results, label=""):
        if not results:
            return f"No events found {label}."

        return f"Events {label}: " + "; ".join(e["event"] for e in results)

    # 🔥 Helper: extract year safely
    def extract_year(self, text):
        match = re.search(r"\d{4}", text)
        return int(match.group()) if match else None

    def answer_query(self, query: str) -> str:
        text = query.lower().strip()

        events = self.memory.events

        if not events:
            return "No events are currently stored in memory."

        # Sort once
        events = sorted(events, key=lambda e: e.get("time", 0))

        year = self.extract_year(text)

        # 🔥 1. Query: "in 2022"
        if year and ("in" in text or "during" in text):
            results = [e for e in events if e.get("time") == year]
            return self.format_results(results, f"in {year}")

        # 🔥 2. Query: "after 2022"
        if year and "after" in text:
            results = [e for e in events if e.get("time", 0) > year]
            return self.format_results(results, f"after {year}")

        # 🔥 3. Query: "before 2022"
        if year and "before" in text:
            results = [e for e in events if e.get("time", 0) < year]
            return self.format_results(results, f"before {year}")

        # 🔥 4. Query: "between 2020 and 2023"
        years = re.findall(r"\d{4}", text)
        if "between" in text and len(years) >= 2:
            y1, y2 = int(years[0]), int(years[1])
            results = [e for e in events if y1 <= e.get("time", 0) <= y2]
            return self.format_results(results, f"between {y1} and {y2}")

        # 🔥 5. Query: "latest event"
        if "latest" in text or "last" in text:
            last_event = events[-1]
            return f"Latest event: {last_event['event']} ({last_event['time']})"

        # 🔥 6. Query: "first event"
        if "first" in text or "earliest" in text:
            first_event = events[0]
            return f"First event: {first_event['event']} ({first_event['time']})"

        # 🔥 7. Query: "what did John do"
        words = text.split()
        for word in words:
            name = word.capitalize()
            subject_events = [e for e in events if e.get("subject") == name]
            if subject_events:
                return self.format_results(subject_events, f"for {name}")

        # 🔥 8. Query: "who got promoted / who joined"
        for e in events:
            if e.get("action") and e["action"] in text:
                filtered = [ev for ev in events if ev.get("action") == e["action"]]
                subjects = list(set(ev["subject"] for ev in filtered))
                return f"{e['action'].capitalize()} done by: " + ", ".join(subjects)

        # 🔥 9. Query: "summary"
        if "summary" in text or "all events" in text:
            return self.format_results(events, "of all stored events")

        # 🔥 fallback
        return "I couldn't understand the query. Try something like 'events after 2022' or 'what did John do'."