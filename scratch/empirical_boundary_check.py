import os
import groq
from dotenv import load_dotenv

load_dotenv()
client = groq.Groq()

test_phrases = [
    "I'd rather you didn't reach out for a while.",
    "Please don't contact me about this again.",
    "We're actually working with someone already.",
    "I prefer not to be contacted anymore.",
    "Please leave me alone.",
    "We already signed with a realtor last week.",
    "Do not reach out to this number.",
    "I'd appreciate it if you don't call this number again.",
    # And benign phrases to ensure zero false positives:
    "So we have been looking at a few different options for a while now, honestly.",
    "We are not in rush exactly, but our lease is up in four months.",
    "I'm not sure if now is the right time for us.",
]

for phrase in test_phrases:
    prompt = (
        'You are an expert sales conversational linguist. Analyze this speaker turn.\n\n'
        'Context (last 2 turns):\n\n'
        f'Current turn (client): "{phrase}"\n\n'
        'Classify into valid JSON with these exact fields:\n'
        "- question_type: 'evaluation' | 'transactional' | 'clarifying' | 'hostile' | 'rhetorical' | 'none'\n"
        "- specificity_score: float 0.0 to 1.0 (dates, dollar amounts, named entities, hard numbers)\n"
        "- future_language_score: float 0.0 to 1.0 (operational future commitment vs vague hypotheticals)\n"
        "- agreement_score: float 0.0 to 1.0 (substantive meeting/pricing commitment ~0.8-1.0; polite nod like 'yeah' ~0.2-0.3)\n"
        "- boundary_score: float 0.0 or 1.0 (STRICT: 1.0 ONLY for explicit stop-contact, DNC, existing broker representation, or legal threats. Exploring options, general hesitation, or reluctance MUST be 0.0)\n\n"
        "Output ONLY raw JSON."
    )
    resp = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150,
        reasoning_effort="none",
    )
    content = resp.choices[0].message.content.strip()
    print(f"Phrase: {phrase}")
    print(f"  -> {content}")
