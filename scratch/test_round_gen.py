import os
import json
import random
import traceback
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')

stages = [
    ("Discovery & Onboarding", "Customer is asking how fast onboarding will take and what implementation assistance will be provided."),
    ("Security & Data Privacy", "Customer is a CISO asking about SOC 2, zero trust, encryption at rest/transit, and cloud isolation."),
    ("Competitive Differentiator", "Customer says they already use an established competitor and asks why they should bother migrating."),
    ("Pricing & Budget Defense", "Customer claims the proposal is 25% higher than their approved quarterly budget and demands justification."),
    ("Adoption & Team Resistance", "Customer is worried their sales reps will resist learning new software and asks about ramp time."),
    ("Executive & CFO Business Case", "Customer asks for the single most compelling revenue acceleration metric to present to their CFO."),
    ("Contract Terms & Flexibility", "Customer asks if you offer a 90-day pilot milestone before locking into a 12-month agreement."),
    ("Closing & Launch Timeline", "Customer is satisfied and asks what exact steps happen immediately after signing the contract today.")
]

for round_num, (stage, context) in enumerate(stages, 1):
    prompt = f"""You are generating dynamic B2B sales simulation content for Round {round_num} of 8.
Sales Stage: {stage}
Context: {context}

Generate:
1. "persona_name": A realistic business buyer full name (e.g. "Sarah Jenkins").
2. "persona_title": Realistic executive job title (e.g. "VP of Operations").
3. "question_text": A realistic, dynamic question or pushback from this buyer (1-2 sentences).
4. "teleprompt_text": An ideal, high-converting salesperson response script to repeat (20-35 words, crisp, confident).

Return strictly JSON with keys: persona_name, persona_title, question_text, teleprompt_text.
"""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an enterprise sales simulator returning valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.75,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        print(f"=== ROUND {round_num}: {stage} ===")
        print(f"Buyer: {data.get('persona_name')} ({data.get('persona_title')})")
        print(f"Q: {data.get('question_text')}")
        print(f"Teleprompt: {data.get('teleprompt_text')}\n")
    except Exception as e:
        print(f"ERROR in round {round_num}:", e)
