import re

EXPANDED_STOPWORDS = {
    'a', 'an', 'the', 'that', 'this', 'these', 'those', 'our', 'your', 'my', 'we', 'you',
    'i', 'us', 'it', 'to', 'for', 'in', 'on', 'at', 'with', 'still', 'way', 'too', 'is',
    'are', 'was', 'were', 'and', 'but', 'or', 'of', 'as', 'be', 'so', 'do', 'does', 'did',
    'just', 'really', 'honestly', 'hear', 'see', 'look', 'what', 'where', 'when', 'why',
    'how', 'here', 'there', 'right', 'well', 'like', 'feel', 'feels', 'about', 'out',
    'then', 'now', 'not', 'no', 'yes', 'yeah', 'ok', 'okay', 'sure', 'get', 'got',
    'can', 'could', 'would', 'should', 'have', 'has', 'had', 'been', 'much', 'more',
    'most', 'some', 'any', 'all', 'very', 'even', 'actually', 'mean', 'think', 'know',
    'something', 'someone', 'everything', 'anything', 'thing', 'things'
}

t13 = "Honestly, a five person commission feels pretty steep for what's involved here."
t14 = "I hear you, but five person still just does not see it right with me."

tok13 = {t for t in re.findall(r'\b[a-z]{3,}\b', t13.lower()) if t not in EXPANDED_STOPWORDS}
tok14 = {t for t in re.findall(r'\b[a-z]{3,}\b', t14.lower()) if t not in EXPANDED_STOPWORDS}
common = tok14.intersection(tok13)
ov_min = len(common) / min(len(tok13), len(tok14)) if common else 0
ov_max = len(common) / max(len(tok13), len(tok14)) if common else 0
print("tok13:", tok13)
print("tok14:", tok14)
print("common:", common)
print("overlap_min:", ov_min, "overlap_max:", ov_max)
