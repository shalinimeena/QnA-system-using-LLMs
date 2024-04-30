import os
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Set the Hugging Face API key
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Step 1: Document Preprocessing
def preprocess_document(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    paragraphs = re.split(r'\n+', text)
    return paragraphs

# Step 2: Answer Generation
def generate_answer(context, query, model_name='NousResearch/Llama-2-7b-hf'):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Perform sentence embeddings
        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = sentence_model.encode([query])[0]
        context_embeddings = sentence_model.encode(context)

        # Find the most relevant sentence based on semantic similarity
        similarity_scores = np.dot(context_embeddings, query_embedding)
        most_relevant_index = np.argmax(similarity_scores)
        most_relevant_sentence = context[most_relevant_index]

        input_text = f"Question: {query}\\\\nContext: {most_relevant_sentence}"
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
        paragraphs = preprocess_document(pdf_file)
        answer = generate_answer(paragraphs, query)
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()