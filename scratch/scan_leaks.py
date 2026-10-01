import json
import glob

matches = []
target_phrase = "Thursday at 3 works, and my wife will be there"

for f in glob.glob("reports/**/*.json", recursive=True):
    try:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        tl = d.get("timeline", [])
        for step in tl:
            sd = step.get("strategic_decision")
            if sd:
                ev = sd.get("evidence_considered", [])
                for e in ev:
                    if target_phrase in e and target_phrase not in step.get("text", ""):
                        matches.append({
                            "file": f,
                            "turn_id": step.get("turn_id"),
                            "text": step.get("text"),
                            "evidence": e,
                            "sd_call_sid": sd.get("call_sid"),
                            "sd_call_id": sd.get("call_id"),
                            "source_state_version": sd.get("source_state_version")
                        })
    except Exception as ex:
        pass

print(f"Total cross-leak matches found: {len(matches)}")
for m in matches[:10]:
    print(m)
