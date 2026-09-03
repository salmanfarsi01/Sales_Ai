import os
import json
import traceback
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()

groq_key = os.getenv('GROQ_API_KEY')
model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')

print("GROQ KEY:", groq_key[:10] if groq_key else "None")
print("GROQ MODEL:", model)

client = Groq(api_key=groq_key)

prompt = """You are a master enterprise sales coach designing an 8-round voice calibration training session.
Target Industry: B2B Enterprise Cloud & AI

Generate exactly 8 sequential sales calibration rounds covering these stages:
Round 1: Discovery and Onboarding (time-to-value, setup complexity)
Round 2: Security and Compliance (SOC 2, data privacy, architecture)
Round 3: Competitive Differentiator (why switch from legacy vendor, AI advantage)
Round 4: Pricing Pushback and ROI (cost justification, 4x ROI metric)
Round 5: Adoption and Change Management (rep usability, low friction)
Round 6: Executive and CFO Justification (strategic revenue impact)
Round 7: Contract Terms and Flexibility (trial milestone, SLA guarantee)
Round 8: Closing and Next Steps (immediate kickoff, workspace launch)

For each round provide:
1. round_number: 1 to 8
2. stage: Stage title
3. persona_name: Realistic buyer name
4. persona_title: Realistic executive title
5. question_text: Natural, realistic buyer question or objection (1-2 sentences).
6. teleprompt_text: Crisp, high-converting salesperson response script to repeat (20-35 words).

Respond ONLY with valid JSON in this exact structure:
{
  "rounds": [
    {
      "round_number": 1,
      "stage": "Discovery and Onboarding",
      "persona_name": "Marcus Vance",
      "persona_title": "VP of Engineering",
      "question_text": "...",
      "teleprompt_text": "..."
    }
  ]
}
"""

try:
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You create high-converting enterprise sales calibration programs in strictly valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    raw = res.choices[0].message.content
    print("RAW LENGTH:", len(raw))
    print("RAW PREVIEW:", raw[:400])
    parsed = json.loads(raw)
    print("PARSED KEYS:", parsed.keys())
    rounds = parsed.get("rounds", [])
    print("ROUNDS COUNT:", len(rounds))
    if rounds:
        print("FIRST ROUND:", rounds[0])
except Exception as e:
    traceback.print_exc()
