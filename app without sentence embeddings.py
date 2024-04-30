import os
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Set the Hugging Face API key
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Step 1: Document Preprocessing
def preprocess_document(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Answer Generation
def generate_answer(context, query, model_name='NousResearch/Llama-2-7b-hf'):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        input_text = f"Question: {query}\\\\nContext: {context}"
        qa_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)
        answer = qa_pipeline(input_text, max_length=1024, num_beams=4, early_stopping=True)[0]['generated_text']
        return answer
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.title("PDF Question Answering")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    query = st.text_input("Enter your question")

    if pdf_file is not None and query:
        pdf_text = preprocess_document(pdf_file)
        answer = generate_answer(pdf_text, query)
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()