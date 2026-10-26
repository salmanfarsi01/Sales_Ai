import re

def extract_proposed_time_or_day(text: str) -> str | None:
    t = text.strip()
    
    # 1. Day + optional window + clock time: "tomorrow afternoon at 3:00 PM", "Thursday at 4", "Friday at 10am"
    m_day_win_time = re.search(
        r"\b((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today))\s*(?:(?:in\s+the\s+|on\s+)?(morning|afternoon|evening))?\s*(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
        t,
        re.IGNORECASE
    )
    if m_day_win_time and m_day_win_time.group(3):
        day_raw = m_day_win_time.group(1).strip()
        day_part = day_raw.lower() if day_raw.lower() in ("tomorrow", "today") else day_raw.capitalize()
        time_part = m_day_win_time.group(3).strip()
        # If no am/pm specified
        if not re.search(r"(?:am|pm)", time_part, re.IGNORECASE):
            num = int(time_part.split(":")[0])
            if m_day_win_time.group(2) and "morning" in m_day_win_time.group(2).lower():
                suffix = "AM"
            else:
                suffix = "PM" if 1 <= num <= 7 else "AM"
            time_part = f"{num}:00 {suffix}" if ":" not in time_part else f"{time_part} {suffix}"
        else:
            time_part = time_part.upper()
            if ":" not in time_part:
                time_part = re.sub(r"(\d+)\s*(AM|PM)", r"\1:00 \2", time_part)
        return f"{day_part} at {time_part}"

    # 2. Day + Window (without clock time): "tomorrow afternoon", "Thursday morning"
    m_day_win = re.search(
        r"\b((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today))\s+(morning|afternoon|evening)\b",
        t,
        re.IGNORECASE
    )
    if m_day_win:
        d = m_day_win.group(1).strip()
        day_part = d.lower() if d.lower() in ("tomorrow", "today") else d.capitalize()
        return f"{day_part} {m_day_win.group(2).lower()}"

    # 3. Standalone clock time: "at 3:00 PM", "at 4"
    m_time = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", t, re.IGNORECASE)
    if m_time:
        time_part = m_time.group(1).strip()
        if not re.search(r"(?:am|pm)", time_part, re.IGNORECASE):
            num = int(time_part.split(":")[0])
            suffix = "PM" if 1 <= num <= 7 else "AM"
            time_part = f"{num}:00 {suffix}" if ":" not in time_part else f"{time_part} {suffix}"
        else:
            time_part = time_part.upper()
            if ":" not in time_part:
                time_part = re.sub(r"(\d+)\s*(AM|PM)", r"\1:00 \2", time_part)
        return f"at {time_part}"

    # 4. Standalone day: "tomorrow", "Thursday"
    m_day = re.search(r"\b((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow))\b", t, re.IGNORECASE)
    if m_day:
        d = m_day.group(1).strip()
        return d.lower() if d.lower() == "tomorrow" else d.capitalize()

    return None

test_cases = [
    "What are your consultation hours? Can we meet Thursday at 4?",
    "Can we meet tomorrow afternoon at 3:00 PM?",
    "Can we do tomorrow at 3pm?",
    "How about tomorrow afternoon?",
    "I can do Thursday at 4.",
    "Can we meet Friday morning?",
    "Let's meet tomorrow",
    "What are your consultation hours?"
]

for tc in test_cases:
    print(f"'{tc}' -> '{extract_proposed_time_or_day(tc)}'")
