import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/recommend_full"  # Change to Render URL after deployment

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🔍 SHL Assessment Recommendation Engine")
st.markdown("Enter a job description or query below, and get personalized SHL assessment recommendations.")

query = st.text_area("Job Description or Query", height=200)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query to proceed.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                response = requests.post(API_URL, json={"query": query})
                if response.status_code == 200:
                    assessments = response.json()
                    if assessments:
                        df = pd.DataFrame(assessments)
                        df["url"] = df["url"].apply(lambda x: f"[Link]({x})")
                        df["description"] = df["description"].str.replace("\n", " ", regex=False)
                        df.rename(columns={
                            "title": "Title",
                            "url": "Assessment Link",
                            "description": "Description",
                            "duration": "Duration (min)",
                            "remote_support": "Remote Support",
                            "adaptive_support": "Adaptive Support",
                            "test_type": "Test Types"
                        }, inplace=True)
                        st.success("Recommendations ready!")
                        st.write(df.to_markdown(index=False), unsafe_allow_html=True)
                    else:
                        st.info("No relevant assessments found.")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.exception(f"Error fetching recommendations: {e}")

st.title("Analysis")
st.markdown(
    "This app uses a **FastAPI** backend and a **Streamlit** frontend. "
    "For ML, it uses **Sentence-BERT** embeddings + **FAISS** for semantic similarity, "
    "and **BM25** for lexical matching. Results are merged using **Reciprocal Rank Fusion (RRF)**, and Google **Gemini 1.5 Flash API** is used to extract data from the given query before processing."
)
st.markdown(
    "The model achieves **0.30 Recall@10** and **0.20 MAP@10** on the provided test dataset, which is limited by the small test dataset, and the lack of any training/fine-tuning of the model. The model is shown to perform well on technical queries, but has high amount of misses on non-technical queries, indicating lack of depth in semantic understanding of the text, which could be improved using ensembling, fine-tuning or even training a classifier model, though effects on latency remain to be observed."
)
st.markdown(
    "Further analysis, as well as the source code can be found in the [GitHub Repository](https://github.com/adityachopra0306/SHL-Recommender) for this project."
)
st.markdown("---")
st.markdown("**Made by [Aditya Chopra](https://www.linkedin.com/in/aditya-chopra-8047582ab/) | [GitHub](https://github.com/adityachopra0306)**")
