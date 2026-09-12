import sys
sys.path.insert(0, ".")
import re

from copilot.behavioral_normalization import normalize_generic_transcript
from copilot.behavioral_semantic import SemanticFeatureEngine

EXPANDED_STOPWORDS = {
    "a", "an", "the", "that", "this", "these", "those", "our", "your", "my", "we", "you",
    "i", "us", "it", "to", "for", "in", "on", "at", "with", "still", "way", "too", "is",
    "are", "was", "were", "and", "but", "or", "of", "as", "be", "so", "do", "does", "did",
    "just", "really", "honestly", "hear", "heard", "see", "seen", "look", "what", "where",
    "when", "why", "how", "here", "there", "right", "well", "like", "feel", "feels",
    "about", "out", "then", "now", "not", "no", "yes", "yeah", "ok", "okay", "sure",
    "get", "got", "can", "could", "would", "should", "have", "has", "had", "been",
    "much", "more", "most", "some", "any", "all", "very", "even", "actually", "mean",
    "think", "know", "something", "someone", "everything", "anything", "thing", "things",
}

turns_data = [
    (2, "So we have been renting downtown", 960, 3860),
    (3, "for about three years,", 4320, 5860),
    (4, "and, honestly, we are just starting to explore what's out there.", 6240, 9860),
    (5, "My partner and I keep going back and forth on whether it is", 10640, 16405),
    (6, "the right time to buy.", 16945, 18325),
    (7, "But", 19345, 19845),
    (8, "our lease renews in five months, so we do need to figure this out soon. We want something with the yard. Ideally, two bedrooms. Somewhere our dog", 20305, 30085),
    (9, "actually has room to", 30500, 31960),
    (10, "run around.", 32260, 33000),
    (11, "We looked at a townhouse two weeks ago that we bought really liked,", 33620, 38280),
    (12, "but the seller ended up taking another offer before we could move on it.", 38660, 43480),
    (13, "Honestly, a five person commission feels pretty steep for what's involved here.", 44855, 50555),
    (14, "I hear you, but five person still just does not see it right with me.", 52615, 57435),
]

engine = SemanticFeatureEngine()
engine.STOPWORDS = EXPANDED_STOPWORDS

# Monkeypatch STOPWORDS in copilot.behavioral_semantic
import copilot.behavioral_semantic
copilot.behavioral_semantic.STOPWORDS = EXPANDED_STOPWORDS

history = []
print("RUNNING RECURRENCE CHECK ACROSS ALL 14 TURNS:")
for idx, text, s_ms, e_ms in turns_data:
    utt = normalize_generic_transcript(text, "client", s_ms, e_ms, "test_call")
    rec_id, rec_type, rec_count = engine.detect_semantic_recurrence(text, history)
    print(f"Turn {idx}: '{text[:40]}...' -> rec_type: {rec_type}, count: {rec_count}, id: {rec_id}")
    history.append(utt)
