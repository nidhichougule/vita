
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DevTrace-NLP", page_icon="🔍")

st.title("🔍 DevTrace-NLP")
st.subheader("NLP-Based Developer Activity Analysis")

st.write(
    "Explore developer commit messages using text preprocessing, "
    "TF-IDF feature extraction, and cosine similarity."
)

commits = st.text_area(
    "Enter commit messages (one per line)",
    value=(
        "Fix login authentication bug\n"
        "Add user registration feature\n"
        "Update documentation and README\n"
        "Resolve database connection issue"
    ),
    height=150,
)

if st.button("Analyze Commits"):
    messages = [line.strip() for line in commits.splitlines() if line.strip()]

    if len(messages) < 2:
        st.warning("Please enter at least two commit messages.")
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(messages)

        similarities = cosine_similarity(matrix)
        results = []

        for i, message in enumerate(messages):
            most_similar = similarities[i].copy()
            most_similar[i] = -1
            match = most_similar.argmax()

            results.append({
                "Commit Message": message,
                "Most Similar Commit": messages[match],
                "Similarity Score": round(float(similarities[i, match]), 3),
            })

        st.success(f"Analyzed {len(messages)} commit messages.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.caption(
            "Similarity scores measure textual similarity, "
            "not bug severity, developer performance, or model accuracy."
        )
