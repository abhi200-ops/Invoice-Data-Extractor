import pandas as pd
import streamlit as st
import pdfplumber
from transformers import pipeline

# Define a function to extract text from a PDF file
def extract_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

# Streamlit App
st.title("Invoice Information Extractor")

# File uploader widget
uploaded_file = st.file_uploader("Upload an invoice PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from the PDF
    extracted_text = extract_text("temp.pdf")
    st.write("Extracted Text from PDF:")
    st.text(extracted_text)

    # Define the questions for extraction
    question1 = "What is the Sales Order No.?"
    question2 = "Give me the Invoice No."

    # Perform question-answering
    sales_order_no = qa_pipeline(question=question1, context=extracted_text)
    invoice_no = qa_pipeline(question=question2, context=extracted_text)

    # Display the results
    st.write(f"Sales Order No: {sales_order_no['answer']}")
    st.write(f"Invoice No: {invoice_no['answer']}")

    # Create a DataFrame to display results in tabular format
    df = pd.DataFrame({
        "Sales Order No": [sales_order_no['answer']],
        "Invoice No": [invoice_no['answer']]
    })

    st.write("Extracted Information:")
    st.dataframe(df)
