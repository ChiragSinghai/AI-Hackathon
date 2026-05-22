import streamlit as st
from rag_pipeline import process_form, ask_question

st.set_page_config(page_title="AI Form Assistant")

st.title("🧠 AI Public Service Form Assistant")

uploaded_file = st.file_uploader(
    "Upload Government Form PDF",
    type=["pdf"]
)

if uploaded_file:

    with open(f"uploads/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully")

    process_form(f"uploads/{uploaded_file.name}")

    query = st.text_input("Ask question about the form")

    if query:
        response = ask_question(query)
        st.write(response)