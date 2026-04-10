#   AI Resume Analysis and Candidate Ranking Tool

This is a Streamlit application that analyzes resumes, ranks candidates based on a job description, and provides a chat interface for querying the resumes. It uses Ollama for local LLM inference.

##   Features

* **Resume Upload:** Upload multiple resumes in PDF or Word format.
* **AI-Powered Analysis:** Generates summaries and insights for each uploaded resume.
* **Candidate Ranking:** Ranks candidates based on a provided job description using Ollama.
* **Chat Interface:** Allows you to ask questions about the uploaded resumes using Ollama.
* **Local LLM:** Uses Ollama for all LLM operations.

##   Prerequisites

* Python 3.8+
* Ollama installed and running
* Ollama model downloaded (e.g., mistral)

##   Installation


1.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Ensure Ollama is running:**

    ```bash
    ollama serve
    ```

3.  **Download the Ollama model (e.g., mistral):**

    ```bash
    ollama pull mistral
    ```

##   Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  Open the Streamlit app in your web browser.

3.  Upload the resume files (PDF or Word).

4.  Enter the Job Description and click "Submit Job Description".

5.  View the summary and insights for each uploaded resume.

6.  View the ranked candidates and their scores.

7.  Use the text input to ask questions about the resumes and click "Send".


