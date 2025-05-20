
import os
import io
import base64
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()
openai = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    images = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text += f"\n--- Page {page_number+1} ---\n"
        text += page.get_text()

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            filename = f"page{page_number+1}_img{img_index+1}.{image_ext}"
            images.append((filename, image))

    return text, images

def chunk_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_relevant_chunk(query, chunks):
    query_embedding = get_embedding(query)
    best_score = -1
    best_chunk = ""
    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)
        score = cosine_similarity(query_embedding, chunk_embedding)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk

def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def ask_about_text_and_image(image, text_context, question):
    image_b64 = encode_image_to_base64(image)
    messages = [
        {"role": "system", "content": "You are an assistant with technical knowledge provided by pdf documentations. Providing ample support, reply in a concise and informative manner."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Here is some text context:\n{text_context}\n\nNow look at the following image and answer:\n{question}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

def main():
    pdf_path = input("Enter the full path to your PDF: ").strip()
    if not os.path.exists(pdf_path):
        print("File not found.")
        return

    print("Reading PDF...")
    text, images = extract_text_and_images(pdf_path)
    chunks = chunk_text(text)

    print(f"\nPDF loaded with {len(chunks)} text chunks and {len(images)} images.")
    print("You can ask questions like 'Explain image 1' or 'What does page 3 describe?'")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if "image" in user_input.lower():
            # Try to find a number in the question to match image index
            import re
            match = re.search(r"image\s*(\d+)", user_input.lower())
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(images):
                    filename, img = images[idx]
                    context_chunk = find_most_relevant_chunk(user_input, chunks)
                    response = ask_about_text_and_image(img, context_chunk, user_input)
                    print(f"\nAssistant (about {filename}):\n{response}\n")
                else:
                    print("Image index out of range.")
            else:
                print("Please specify which image (e.g., 'Explain image 2')")
        else:
            context_chunk = find_most_relevant_chunk(user_input, chunks)
            messages = [
                {"role": "system", "content": "Answer based only on the provided document text."},
                {"role": "user", "content": f"Context: {context_chunk}\n\nQuestion: {user_input}"}
            ]
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            print(f"\nAssistant:\n{response.choices[0].message.content}\n")

if __name__ == "__main__":
    main()