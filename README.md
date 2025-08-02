https://chat-with-any-pdf-lk1612kselmp83kt5gcvc4.streamlit.app/   (Deployment Link)

# Chat with Your PDF 

This is a simple web application built with Streamlit and LangChain that allows you to upload a PDF document and ask questions about its content. The app uses Google's Gemini 1.5 Flash model to understand the document and provide intelligent answers.

##  Features

-   **Upload PDFs**: Easily upload any PDF document directly through the web interface.
-   **Conversational Q&A**: Ask questions in natural language and get answers based on the document's content.
-   **Chat History**: Remembers the conversation, allowing for follow-up questions.
-   **Powered by Gemini**: Leverages the speed and power of Google's Gemini 1.5 Flash model for fast and accurate responses.

---

##  Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8+
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    -   Create a file named `.env` in the root of your project directory.
    -   Add your Google API key to this file:
        ```
        GOOGLE_API_KEY="your_google_api_key_here"
        ```

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application should now be running in your web browser!

---

##  Tech Stack

-   **Framework**: Streamlit
-   **LLM Orchestration**: LangChain
-   **Language Model**: Google Gemini 1.5 Flash
-   **Embeddings**: GoogleGenerativeAIEmbeddings
-   **Vector Store**: FAISS

---

##  Deployment

This application is deployed on **Streamlit Community Cloud**. Any changes pushed to the `main` branch of the GitHub repository will trigger an automatic redeployment.
