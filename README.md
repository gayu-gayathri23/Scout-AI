# TalentScout AI 🎯
AI-Powered Talent Scouting & Engagement Agent — Deccan AI Catalyst Hackathon

## What it does
Takes a Job Description → discovers matching candidates → simulates recruiter outreach conversations → outputs a ranked shortlist scored on two dimensions: Match Score and Interest Score.

## Architecture
1. JD Parser — Claude extracts structured requirements (skills, experience, responsibilities)
2. Candidate Discovery — Vector search using FAISS + SentenceTransformers over candidate profiles
3. Matching Engine — LLM scoring with explainability per candidate
4. Outreach Agent — Simulated 3-turn recruiter conversation, LLM plays the candidate persona
5. Ranking Engine — Final score = Match × 0.6 + Interest × 0.4

## Scoring logic
| Dimension | Weight | How computed |
|-----------|--------|--------------|
| Match Score | 60% | LLM evaluates skills, experience, role fit |
| Interest Score | 40% | LLM scores interest from simulated conversation |

## Setup
pip install -r requirements.txt
Add ANTHROPIC_API_KEY to .env file
streamlit run app.py

## Tech Stack
- LLM: Claude Sonnet (Anthropic)
- Embeddings: SentenceTransformers all-MiniLM-L6-v2
- Vector Search: FAISS
- UI: Streamlit
- Hosting: Streamlit Cloud