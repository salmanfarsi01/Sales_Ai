import re

with open("web/conversation_state_replay.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50\u2b55]")

print("=== EMOJI OCCURRENCES ===")
for i, line in enumerate(lines, 1):
    found = emoji_pattern.findall(line)
    if found:
        # print line number and repr of found
        safe_line = line.strip().encode("ascii", "replace").decode("ascii")
        print(f"Line {i}: {[f.encode('unicode_escape').decode('ascii') for f in found]} -> {safe_line}")

print("\n=== BTNSHARELINK OCCURRENCES ===")
for i, line in enumerate(lines, 1):
    if "btnShareLink" in line or "Share Link" in line:
        safe_line = line.strip().encode("ascii", "replace").decode("ascii")
        print(f"Line {i}: {safe_line}")
