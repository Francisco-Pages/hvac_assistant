import streamlit as st
import fitz
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load environment
load_dotenv()
openai = OpenAI()

# PDF extraction
def extract_text_and_images(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    images = []

    for page_num, page in enumerate(doc):
        text += f"\n--- Page {page_num+1} ---\n"
        text += page.get_text()

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"]))
            images.append((f"Page {page_num+1} Image {img_index+1}", image))

    return text, images

def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# --- Utility function to truncate long context text ---
def truncate_text(text, max_tokens=30000):
    enc = tiktoken.encoding_for_model("gpt-4")  # works for gpt-4o and gpt-4-turbo too
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def ask_gpt4o_with_image(image, context, question):
    image_b64 = encode_image_to_base64(image)
    truncated_context = truncate_text(context)
    messages = [
        {"role": "system", "content": "You are a document assistant that explains images and text."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Context: {truncated_context}\n\nQuestion: {question}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

def ask_gpt4o_text_only(context, question):
    truncated_context = truncate_text(context)
    messages = [
        {"role": "system", "content": "Answer based only on the following document text."},
        {"role": "user", "content": f"Context: {truncated_context}\n\nQuestion: {question}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

# --- Streamlit App ---

st.set_page_config(page_title="PDF Chatbot with Vision", layout="wide")
st.title("🧠 PDF + Image Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with st.spinner("Extracting content..."):
        text, images = extract_text_and_images(uploaded_file)

    st.success("PDF processed!")
    question = st.text_input("Ask a question about the document:")

    if question:
        col1, col2 = st.columns([2, 1])
        with col1:
            response = ask_gpt4o_text_only(text, question)
            st.markdown("#### 💬 Answer:")
            st.write(response)

        with col2:
            st.markdown("#### 🖼️ Available Images:")
            for idx, (label, img) in enumerate(images):
                if st.button(f"Explain {label}", key=f"btn_{idx}"):
                    answer = ask_gpt4o_with_image(img, text, question)
                    st.image(img, caption=label, use_column_width=True)
                    st.write("🔍 GPT-4o says:")
                    st.write(answer)