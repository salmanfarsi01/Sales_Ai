import os
import groq
from dotenv import load_dotenv

load_dotenv()
client = groq.Groq()

prompt = (
    'You are an expert sales conversational linguist. Analyze this speaker turn.\n\n'
    'Context (last 2 turns):\n\n'
    'Current turn (client): "So we have been looking at a few different options for a while now, honestly."\n\n'
    'Classify into valid JSON with these exact fields:\n'
    "- question_type: 'evaluation' | 'transactional' | 'clarifying' | 'hostile' | 'rhetorical' | 'none'\n"
    "- specificity_score: float 0.0 to 1.0 (dates, dollar amounts, named entities, hard numbers)\n"
    "- future_language_score: float 0.0 to 1.0 (operational future commitment vs vague hypotheticals)\n"
    "- agreement_score: float 0.0 to 1.0 (substantive meeting/pricing commitment ~0.8-1.0; polite nod like 'yeah' ~0.2-0.3)\n"
    "- boundary_score: float 0.0 to 1.0 (stop contact, already represented, privacy constraints)\n\n"
    "Output ONLY raw JSON."
)

resp = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=150,
    reasoning_effort="none",
)
print("Groq Response:")
print(resp.choices[0].message.content)
