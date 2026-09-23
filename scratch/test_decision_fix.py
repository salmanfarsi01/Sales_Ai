import sys
sys.path.insert(0, '.')
import json
import re
from copilot.conversation_state_manager import ConversationStateManager
from copilot.conversation_state_contract import BehavioralSignalInputBundle

with open('reports/synthetic/conversation_state_sim_mucj0p5s.json') as f:
    data = json.load(f)

mgr = ConversationStateManager(call_sid='test_call')
eb1 = BehavioralSignalInputBundle(**data['timeline'][0]['evidence_bundle'])
mgr.process_turn_bundle(eb1)
eb2 = BehavioralSignalInputBundle(**data['timeline'][1]['evidence_bundle'])

text = eb2.utterance_text.lower().strip()
patterns = [
    r"\b(?:i'm|i\s+am)\s+the\s+(?:one|sole\s+person)\s+(?:making|who\s+makes)\b",
    r"\bno\s+one\s+else\s+needs?\s+to\s+sign\b",
    r"\bi\s+make\s+the\s+decisions?\s+alone\b",
]
print("Turn 2 text:", text)
print("Matches regex:", any(re.search(p, text) for p in patterns))
