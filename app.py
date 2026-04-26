import streamlit as st
import pandas as pd
import json
from agent import run_pipeline

st.set_page_config(
    page_title="TalentScout AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 TalentScout AI")
st.caption("AI-Powered Talent Scouting & Engagement Agent — Deccan AI Catalyst Hackathon")

SAMPLE_JD = """We are looking for a Senior AI/ML Engineer to join our growing team.

Requirements:
- 4+ years of experience in machine learning and AI
- Strong Python skills and experience with LLMs
- Experience with LangChain, RAG systems, or agent frameworks
- Familiarity with vector databases (Pinecone, FAISS, Chroma)
- Experience deploying ML models to production

Nice to have:
- Experience with fine-tuning LLMs
- Knowledge of Hugging Face ecosystem
- Open source contributions

Responsibilities:
- Build and maintain LLM-powered product features
- Design and implement RAG pipelines
- Work closely with product and engineering teams
- Evaluate and integrate new AI tools and models"""

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Job Description")
    jd = st.text_area("Paste your JD here", value=SAMPLE_JD, height=350)
    top_k = st.slider("Number of candidates to scout", min_value=3, max_value=8, value=5)
    run_btn = st.button("🚀 Run Talent Scout", type="primary", use_container_width=True)

with col2:
    st.subheader("🔍 Parsed Requirements")
    parsed_placeholder = st.empty()

if run_btn:
    if not jd.strip():
        st.error("Please enter a job description.")
    else:
        st.divider()
        status = st.status("Running TalentScout AI pipeline...", expanded=True)
        results_placeholder = st.empty()
        all_candidates = []

        try:
            for event_type, data in run_pipeline(jd, top_k=top_k):
                if event_type == "step":
                    status.write(f"⚡ {data}")

                elif event_type == "jd":
                    with parsed_placeholder.container():
                        st.json(data)

                elif event_type == "candidate":
                    all_candidates.append(data)
                    df = pd.DataFrame([{
                        "Name": c["name"],
                        "Title": c["title"],
                        "Match Score": c.get("match_score", "-"),
                        "Interest Score": c.get("interest_score", "..."),
                        "Interest Level": c.get("interest_level", "..."),
                    } for c in all_candidates])
                    results_placeholder.dataframe(df, use_container_width=True)

                elif event_type == "final":
                    status.update(label="✅ Scouting complete!", state="complete", expanded=False)
                    results_placeholder.empty()

                    st.subheader("🏆 Final Ranked Shortlist")

                    for rank, c in enumerate(data, 1):
                        with st.expander(
                            f"#{rank} {c['name']} — Final Score: {c['final_score']}/100 | Match: {c['match_score']}% | Interest: {c['interest_score']}%",
                            expanded=(rank <= 3)
                        ):
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Match Score", f"{c['match_score']}%")
                            col_b.metric("Interest Score", f"{c['interest_score']}%")
                            col_c.metric("Final Score", f"{c['final_score']}%")

                            st.write(f"**Title:** {c['title']} | **Experience:** {c['years_exp']} years | **Location:** {c['location']}")
                            st.write(f"**Skills:** {', '.join(c['skills'])}")

                            st.write("**Why they match:**")
                            st.info(c.get('match_explanation', ''))

                            if c.get('skill_gaps'):
                                st.warning(f"**Skill gaps:** {', '.join(c['skill_gaps'])}")

                            st.write(f"**Interest signal:** {c.get('interest_signal', '')}")

                            with st.expander("💬 View simulated conversation"):
                                for turn in c.get('conversation', []):
                                    st.write(f"🤝 **Recruiter:** {turn['q']}")
                                    st.write(f"👤 **{c['name'].split()[0]}:** {turn['a']}")
                                    st.divider()

                    st.subheader("📥 Export Results")
                    export_data = [{
                        "rank": i + 1,
                        "name": c["name"],
                        "title": c["title"],
                        "match_score": c.get("match_score"),
                        "interest_score": c.get("interest_score"),
                        "final_score": c.get("final_score"),
                        "interest_level": c.get("interest_level"),
                        "match_explanation": c.get("match_explanation"),
                        "skill_gaps": c.get("skill_gaps")
                    } for i, c in enumerate(data)]

                    col_d, col_e = st.columns(2)
                    with col_d:
                        st.download_button(
                            "⬇️ Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name="talent_shortlist.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    with col_e:
                        df_export = pd.DataFrame(export_data)
                        st.download_button(
                            "⬇️ Download CSV",
                            data=df_export.to_csv(index=False),
                            file_name="talent_shortlist.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Check your GEMINI_API_KEY in the .env file and try again.")