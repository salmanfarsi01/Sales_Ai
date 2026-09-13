import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratch.test_play_by_play_suite import run_client_play_by_play
import io
from contextlib import redirect_stdout

async def main():
    f = io.StringIO()
    with redirect_stdout(f):
        await run_client_play_by_play()
    raw = f.getvalue()
    turns = json.loads(raw)

    headers = ["Turn", "Time (s)", "Transcript Excerpt", "WPM", "Base", "Pace Delta", "Latency", "Question?", "Recurrence", "Bound.", "Spec.", "Trust", "Ready"]
    rows = []
    for t in turns:
        trunc_text = t["text"][:50] + "..." if len(t["text"]) > 50 else t["text"]
        rec = t["recurrence"] if t["recurrence"] != "none" else "none"
        q = t["question"] if t["question"] != "none" else "none"
        lat = f"{t['latency_ms']}ms" if t["latency_ms"] != "\u2014" else "\u2014"
        base = f"{t['baseline_wpm']}" if t["baseline_wpm"] else "\u2014"
        rows.append([
            f"#{t['turn']}",
            t["time_sec"],
            trunc_text,
            str(t["wpm"]),
            base,
            t["pace_delta_pct"],
            lat,
            q,
            rec,
            str(t["boundary"]),
            str(t["specificity"]),
            f"{t['trust']}%",
            f"{t['readiness']}%",
        ])

    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_str = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    print(header_str)
    print(sep_str)
    for r in rows:
        print("| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(r)) + " |")

if __name__ == "__main__":
    asyncio.run(main())
