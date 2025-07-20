import streamlit as st
import requests
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Document Q&A with RAG", layout="wide")

# --- UI Components ---
st.title("RAG Implementation Project")
st.markdown("Upload your documents, ask questions, and get answers from the content.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Embed All Files"):
            with st.spinner("Processing files... This may take a moment."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    try:
                        response = requests.post(f"{API_URL}/upload", files=files)
                        if response.status_code == 200:
                            st.success(f"✅ Successfully embedded {file.name}")
                        else:
                            st.error(f"❌ Error with {file.name}: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Connection error with {file.name}: {e}")
            st.success("All files processed!")

# --- Main Chat Interface ---
# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        try:
            response = requests.post(f"{API_URL}/query", json={"query": prompt})
            if response.status_code == 200:
                full_response = response.json().get("response", "Sorry, I couldn't find an answer.")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.markdown(f"Error: {response.text}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {response.text}"})
        except requests.exceptions.RequestException as e:
            message_placeholder.markdown(f"Connection Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Connection Error: {e}"}) 