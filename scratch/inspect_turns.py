import sqlite3
import json

conn = sqlite3.connect("knowledge/evidence_log.db")
c = conn.cursor()
c.execute(
    "SELECT timestamp_ms, snapshot_json FROM evidence_snapshots "
    "WHERE call_sid = 'call_29d924fc' AND horizon = 'current_utterance' "
    "ORDER BY timestamp_ms ASC"
)
for t, snap_str in c.fetchall():
    data = json.loads(snap_str)
    sem = data.get("semantic_features", {})
    utt_id = sem.get("utterance_id")
    print(f"Timestamp: {t}ms | Utt: {utt_id}")
    print(f"  Boundary: {sem.get('boundary_score')} | Future: {sem.get('future_language_score')} | Spec: {sem.get('specificity_score')} | Agree: {sem.get('agreement_score')}")
    print(f"  Extraction Mode: {sem.get('extraction_mode')}")
