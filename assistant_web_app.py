import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io

# Load environment
load_dotenv()
openai = OpenAI()

# PDF extraction
def extract_text_and_images(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    images = []

    for i, page in enumerate(doc):
        text += f"\n--- Page {i + 1} ---\n"
        text += page.get_text()

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)

    return text, images

# Ask GPT-4o with optional image
def ask_gpt4o(context, question, images=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers based on the document's content and diagrams."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    files = []
    if images:
        # Attach the first image (you can improve this logic)
        image_file = openai.files.create(file=io.BytesIO(images[0]), purpose="vision")
        files.append(image_file.id)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tool_choice="auto",
            max_tokens=1000,
            temperature=0.2,
            images=[{"image": image_file.id, "detail": "high"}]
        )
    else:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.2
        )

    return response.choices[0].message.content

# --- Streamlit App ---

st.set_page_config(page_title="Chill Bot: Chat Assistant", layout="centered")
st.title("📄🧠 ChillBot: Chat Assistant")

if "context" not in st.session_state:
    st.session_state.context = ""
if "images" not in st.session_state:
    st.session_state.images = []
if "chat" not in st.session_state:
    st.session_state.chat = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting content..."):
        text, images = extract_text_and_images(uploaded_file)
        st.session_state.context = text
        st.session_state.images = images

    st.success("PDF processed! Ask away.")

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Generating answer..."):
            answer = ask_gpt4o(st.session_state.context, question, st.session_state.images)
            st.session_state.chat.append(("You", question))
            st.session_state.chat.append(("Assistant", answer))

    st.markdown("### 💬 Conversation")
    for speaker, msg in st.session_state.chat:
        st.markdown(f"**{speaker}:** {msg}")