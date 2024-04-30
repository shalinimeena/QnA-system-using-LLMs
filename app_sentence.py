import os
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
        input_text = f"Question: {query}\\nContext: {context}"
        print(f"Input Text: {input_text}")  # Print the input text for debugging
        qa_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)
        
        answer = qa_pipeline(input_text, max_length=1024, num_beams=4, early_stopping=True)[0]['generated_text']
        return answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
pdf_file = 'input.pdf'
pdf_text = preprocess_document(pdf_file)
print(pdf_text)
#query = "What are the key factors affecting climate change?"
answer = generate_answer(pdf_text, query, model_name='NousResearch/Llama-2-7b-hf')
print(f"Answer: {answer}")