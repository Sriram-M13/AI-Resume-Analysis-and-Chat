import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

# Ollama model name
OLLAMA_MODEL = "mistral"


def chat_with_ai(vectorstore, query, chat_history):
    """Chats with AI using RAG."""
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = "System: You are a helpful assistant that analyzes resumes and compares them. Use the provided context to answer the user's questions.\n"
    for role, content in chat_history:
        if role == "user":
            prompt += f"User: {content}\n"
        else:
            prompt += f"Assistant: {content}\n"
    prompt += f"Context: {context}\nUser: {query}"

    chat = Ollama(model=OLLAMA_MODEL)
    response = chat(prompt)
    return response


def generate_summary_and_insights(resume_text):
    """Generates a summary and insights for a resume."""
    llm = Ollama(model=OLLAMA_MODEL)
    prompt = f"""
    Please generate a concise summary and key insights for the following resume text:

    {resume_text}

    Summary:
    Insights:
    """
    response = llm(prompt)
    return response


def rank_candidates(resume_data, job_description):
    """Ranks candidates based on job description."""
    llm = Ollama(model=OLLAMA_MODEL)
    candidate_scores = {}
    for file_name, data in resume_data.items():
        resume_text = data["resume_text"]
        prompt = f"""
        Given the following job description:
        {job_description}

        And the following resume:
        {resume_text}

        Provide a score (out of 100) indicating how well the candidate's experience and skills match the job description.
        Return ONLY the score as a single number, with no other words or characters.
        """
        response = llm(prompt)
        try:
            score = int(response.strip())
            candidate_scores[file_name] = {"score": score}
        except ValueError:
            candidate_scores[file_name] = {"score": 0}
    ranked_candidates = sorted(
        candidate_scores.items(), key=lambda item: item[1]["score"], reverse=True
    )
    return ranked_candidates


def main():
    st.title("Resume Analysis and Candidate Ranking (Ollama)")

    # Initialize session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = {}
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "combined_vectorstore" not in st.session_state:
        st.session_state.combined_vectorstore = None
    if "ranked_candidates" not in st.session_state:
        st.session_state.ranked_candidates = [] #store ranked candidates

    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True
    )
    st.session_state.job_description = st.text_area(
        "Enter the Job Description:", value=st.session_state.job_description
    )
    submit_job_desc = st.button("Submit Job Description")

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.resume_data = {}
        for uploaded_file in uploaded_files:
            vectorstore, resume_text = load_and_process_resume(uploaded_file)
            if vectorstore:
                summary_insights = generate_summary_and_insights(
                    resume_text
                )  # Generate summary
                st.session_state.resume_data[uploaded_file.name] = {
                    "vectorstore": vectorstore,
                    "resume_text": resume_text,
                    "summary_insights": summary_insights,
                }

        # Combine vectorstores
        if st.session_state.resume_data:
            combined_vectorstore = list(st.session_state.resume_data.values())[0][
                "vectorstore"
            ]
            for data in list(st.session_state.resume_data.values())[1:]:
                combined_vectorstore.merge_from(data["vectorstore"])
            st.session_state.combined_vectorstore = combined_vectorstore
        else:
            st.session_state.combined_vectorstore = FAISS.from_documents(
                [], OllamaEmbeddings(model=OLLAMA_MODEL)
            )

        # Rank candidates
        st.session_state.ranked_candidates = rank_candidates(
            st.session_state.resume_data, st.session_state.job_description
        )

    # Display summaries, ranked candidates, and chat
    if st.session_state.resume_data: # only display if  resumes are uploaded
        for file_name, data in st.session_state.resume_data.items():
            st.subheader(f"Summary and Insights for {file_name}")
            st.write(data["summary_insights"])

        """if st.session_state.ranked_candidates: # only display if ranked_candidates is not empty
            st.subheader("Ranked Candidates:")
            for file_name, candidate_info in st.session_state.ranked_candidates:
                st.write(f"**{file_name}: Score = {candidate_info['score']}**")"""

    query = st.text_input("Ask a question about the resumes:")
    if st.button("Send"):
        if query:
            response = chat_with_ai(
                st.session_state.combined_vectorstore, query, st.session_state.chat_history
            )
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", response))
            st.write(f"**You:** {query}")
            st.write(f"**AI:** {response}")

    if st.session_state.chat_history:
        for role, content in st.session_state.chat_history:
            if role == "user":
                st.write(f"**You:** {content}")
            else:
                st.write(f"**AI:** {content}")


def load_and_process_resume(uploaded_file):
    """Loads and processes a single resume."""
    import tempfile
    import os
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OllamaEmbeddings

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_file.name)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_file.name)
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return None, None

    documents = loader.load()
    raw_text = "\n".join([doc.page_content for doc in documents])  # Get raw text
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore, raw_text


if __name__ == "__main__":
    main()
