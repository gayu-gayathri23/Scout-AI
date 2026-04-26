import json
import os
import time
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def load_candidates():
    with open("candidates.json") as f:
        return json.load(f)

def parse_jd(job_description: str) -> dict:
    prompt = f"""Parse this job description and extract key requirements as JSON.
Return ONLY valid JSON with no extra text, no markdown, no backticks. Just raw JSON.
Fields needed:
- role (string): job title
- required_skills (list): must-have technical skills
- nice_to_have (list): optional skills
- min_years_exp (int): minimum years of experience
- key_responsibilities (list): 3-5 main responsibilities

Job Description:
{job_description}"""

    text = call_llm(prompt)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)

def build_candidate_profiles(candidates: list) -> list:
    profiles = []
    for c in candidates:
        profile_text = f"{c['title']} with {c['years_exp']} years experience. Skills: {', '.join(c['skills'])}. {c['summary']}"
        profiles.append(profile_text)
    return profiles

def find_matching_candidates(jd_parsed: dict, candidates: list, top_k: int = 5):
    profiles = build_candidate_profiles(candidates)
    candidate_embeddings = embedder.encode(profiles, convert_to_numpy=True)

    dimension = candidate_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(candidate_embeddings)
    index.add(candidate_embeddings)

    jd_text = f"{jd_parsed['role']}. Skills: {', '.join(jd_parsed['required_skills'])}. {' '.join(jd_parsed['key_responsibilities'])}"
    jd_embedding = embedder.encode([jd_text], convert_to_numpy=True)
    faiss.normalize_L2(jd_embedding)

    scores, indices = index.search(jd_embedding, min(top_k * 2, len(candidates)))

    results = []
    for i, idx in enumerate(indices[0]):
        candidate = candidates[idx]
        vector_score = float(scores[0][i])

        prompt = f"""Rate this candidate's match for the role.
Return ONLY raw JSON with no markdown, no backticks, no extra text.

Role: {jd_parsed['role']}
Required skills: {jd_parsed['required_skills']}
Min experience: {jd_parsed['min_years_exp']} years

Candidate: {candidate['name']}
Title: {candidate['title']}
Skills: {candidate['skills']}
Experience: {candidate['years_exp']} years
Summary: {candidate['summary']}

Return exactly this JSON structure:
{{"match_score": 75, "explanation": "2 sentence explanation here", "skill_gaps": ["gap1", "gap2"]}}"""

        text = call_llm(prompt)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            llm_result = json.loads(text)
        except:
            llm_result = {"match_score": 50, "explanation": "Could not parse.", "skill_gaps": []}

        results.append({
            **candidate,
            "match_score": llm_result.get("match_score", 50),
            "match_explanation": llm_result.get("explanation", ""),
            "skill_gaps": llm_result.get("skill_gaps", []),
            "vector_similarity": round(vector_score * 100, 1)
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_k]

def simulate_outreach(candidate: dict, jd_parsed: dict) -> dict:
    opener = f"Hi {candidate['name'].split()[0]}! I came across your profile and think you'd be a great fit for a {jd_parsed['role']} role. The position involves {', '.join(jd_parsed['key_responsibilities'][:2])}. Would you be open to learning more?"

    questions = [
        opener,
        f"That's great! The role requires {', '.join(jd_parsed['required_skills'][:3])}. How much of your recent work has involved these areas?",
        f"We're looking for someone who can start relatively soon. Is this something you'd be actively considering?"
    ]

    interest_signals = []

    for q in questions:
        persona_prompt = f"""You are {candidate['name']}, a {candidate['title']} with {candidate['years_exp']} years of experience.
Your skills: {', '.join(candidate['skills'])}.
Background: {candidate['summary']}

A recruiter just said to you: "{q}"

Reply naturally as this person in a professional conversation. Show realistic interest based on how well the role fits your background. Write only 2-3 sentences. No labels, just your reply."""

        reply = call_llm(persona_prompt)
        interest_signals.append(reply)

    conversation_text = "\n".join([
        f"Q: {questions[i]}\nA: {interest_signals[i]}" for i in range(len(questions))
    ])

    interest_prompt = f"""Based on this candidate's responses to recruiter outreach, rate their interest level.
Return ONLY raw JSON with no markdown, no backticks, no extra text.

Candidate: {candidate['title']}, Skills: {candidate['skills']}
Role: {jd_parsed['role']}

Conversation:
{conversation_text}

Return exactly this JSON structure:
{{"interest_score": 75, "interest_level": "High", "key_signal": "one sentence about what signals their interest"}}

interest_level must be exactly one of: High, Medium, Low"""

    text = call_llm(interest_prompt)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        interest_data = json.loads(text)
    except:
        interest_data = {"interest_score": 50, "interest_level": "Medium", "key_signal": "Could not assess."}

    return {
        "interest_score": interest_data.get("interest_score", 50),
        "interest_level": interest_data.get("interest_level", "Medium"),
        "interest_signal": interest_data.get("key_signal", ""),
        "conversation": [{"q": questions[i], "a": interest_signals[i]} for i in range(len(questions))]
    }

def compute_final_ranking(candidates_with_scores: list) -> list:
    for c in candidates_with_scores:
        match = c.get("match_score", 0)
        interest = c.get("interest_score", 50)
        c["final_score"] = round(match * 0.6 + interest * 0.4, 1)
    return sorted(candidates_with_scores, key=lambda x: x["final_score"], reverse=True)

def run_pipeline(job_description: str, top_k: int = 5):
    candidates = load_candidates()

    yield "step", "Parsing job description..."
    jd_parsed = parse_jd(job_description)
    yield "jd", jd_parsed

    yield "step", "Finding matching candidates..."
    matched = find_matching_candidates(jd_parsed, candidates, top_k=top_k)

    results = []
    for i, candidate in enumerate(matched):
        yield "step", f"Simulating outreach with {candidate['name']} ({i+1}/{len(matched)})..."
        outreach = simulate_outreach(candidate, jd_parsed)
        candidate.update(outreach)
        results.append(candidate)
        yield "candidate", candidate

    yield "step", "Computing final rankings..."
    final = compute_final_ranking(results)
    yield "final", final
