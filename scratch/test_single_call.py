import os
import traceback
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')

prompt = """You are an elite B2B sales simulation coach generating Round 1 of 8.
Target Industry: B2B Enterprise SaaS & AI
Sales Stage: Discovery & Onboarding
Key Focus: Time to value, setup complexity, and dedicated kickoff support.
Buyer Persona: Rachel (VP of Engineering, Calm & Professional)

Generate:
1. "question_text": A realistic, natural question or objection from Rachel (VP of Engineering). Must be conversational, specific to B2B Enterprise SaaS & AI, and 1-2 sentences.
2. "teleprompt_text": An ideal, high-converting salesperson response script to repeat (20-35 words, confident, structured, addressing Rachel's concern directly).

Respond ONLY with valid JSON with keys "question_text" and "teleprompt_text":
{
  "question_text": "...",
  "teleprompt_text": "..."
}
"""

try:
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an enterprise sales simulator returning valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    print("SUCCESS:", res.choices[0].message.content)
except Exception as e:
    traceback.print_exc()
