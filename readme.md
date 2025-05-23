# PDF Assistant

A Streamlit-powered web application that allows users to upload PDFs and ask questions about their content—including diagrams and technical images. It leverages OpenAI’s GPT-4o to analyze both text and images from documents.

---

## Features

- Upload and extract content from PDF files
- Ask natural language questions about the document's text
- Understand and explain embedded images (diagrams, schematics, etc.)
- Streamlit frontend for easy interaction

---

## Tech Stack

- **Python**
- **Streamlit** — Frontend interface
- **PyMuPDF (fitz)** — PDF parsing and image extraction
- **Pillow (PIL)** — Image processing
- **OpenAI GPT-4o** — Text + Vision model for answering questions
- **tiktoken** — Token management for large document inputs
- **dotenv** — Secure environment variable loading

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Francisco-Pages/hvac_assistant.git
cd hvac_assistant
```
2.	**Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3.	**Set up your OpenAI API key:**
Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key_here
```


⸻

Usage
1.	Run the Streamlit app:
```bash
streamlit run assistant_web_app.py
```

2.	Upload a PDF document.
3.	Ask a question about the text or select an image to get visual explanation.

⸻

Example Use Cases
	•	Ask “What does the schematic on page 4 represent?”
	•	Get summaries of installation instructions from technical manuals
	•	Understand visual content like wiring diagrams or airflow charts

⸻

Limits
	•	Be mindful of token limits. Long documents are automatically truncated to fit the model’s capacity.
	•	The current model (gpt-4o-mini) supports vision and is optimized for performance, but responses may vary with image complexity.

⸻

Contributing

Feel free to open issues or submit pull requests to improve the project. Contributions are welcome!

⸻

License

This project is licensed under the MIT License.

