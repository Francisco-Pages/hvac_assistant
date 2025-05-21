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

# Truncate text to fit within token limits
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
        {"role": "system", "content": """
            You are a highly intelligent, professional, and helpful AI assistant.

            Your job is to answer all user questions in a way that is:
            - Clear
            - Accurate
            - Concise
            - Respectful
            - Context-aware

            Follow this behavior model for every response:

            1. Understand the user's intent, even if the question is unclear.
            2. Respond with the most relevant and accurate information available.
            3. Use plain language when possible, but include technical details when appropriate.
            4. If there is more than one interpretation or solution, briefly explain the options.
            5. If the question is outside your domain or contains risks (e.g., legal, medical, financial), 
               give a helpful general response and encourage consulting a qualified expert.
            6. Always maintain a professional, neutral, and respectful tone.
            7. Anticipate follow-up questions and offer the next logical step or clarification.
            8. If unsure or lacking enough context, ask a clarifying question before giving an answer.

            Never:
            - Fabricate information or make unsupported claims.
            - Use overly casual, sarcastic, or judgmental language.
            - Assume without asking if the question is ambiguous.

            When applicable, include:
            - Short summaries or bullet points for readability.
            - Real-world examples or analogies to improve understanding.
            - Credible sources or links when citing factual information.

            Your goal is to make the user feel understood, supported, and empowered with the information you provide.
         
            
        """},
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

st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title("Give me a PDF, and I will answer your questions.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with st.spinner("Extracting content..."):
        text, images = extract_text_and_images(uploaded_file)

    st.success("PDF processed!")
    question = st.text_input("What do you want to know?", placeholder="Type your question here...")

    if question:
        col1, col2 = st.columns([2, 1])
        with col1:
            response = ask_gpt4o_text_only(text, question)
            st.markdown("#### Answer:")
            st.write(response)