import streamlit as st
from chatbot import IEMSCourseChatbot

st.set_page_config(page_title="IEMS Course Chatbot", layout="wide")

# Initialize Chatbot
@st.cache_resource(show_spinner=False)
def load_chatbot():
    return IEMSCourseChatbot()

chatbot = load_chatbot()

# UI
st.title("🧠 IEMS Course Chatbot")
st.markdown("Ask anything about IEMS courses — prerequisites, what they satisfy, or course descriptions.")

query = st.text_input("🔍 Enter your question", placeholder="e.g., What are the prerequisites for IEMS 302?")

if query and len(query.strip()) >= 3:
    if st.button("Submit"):
        with st.spinner("Processing..."):
            response = chatbot.respond_to_query(query)
        st.markdown("### 📘 Response")
        st.success(response)
else:
    st.info("Type your query above to get started.")

# Examples
with st.expander("💡 Example Queries"):
    st.markdown("""
    - What are the prerequisites for IEMS 302?
    - What does IEMS 304 satisfy?
    - Describe IEMS 303.
    - Which course is easier: IEMS 303 or 304?
    """)
