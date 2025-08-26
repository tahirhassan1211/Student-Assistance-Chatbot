import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import textwrap
import os
from dotenv import load_dotenv
load_dotenv()


# Initialize LLM
llm = ChatGroq(
    model="llama3-8b-8192"
   
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("📘 RAG-Based Student Assistant for Textbooks")

# Step 1: Upload PDF
pdf_file = st.file_uploader("📤 Upload your textbook (PDF)", type=["pdf"])
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    # Use Document Loader
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings and FAISS Vector Store
    #embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    st.success("✅ Textbook uploaded and processed successfully.")

    # Step 2: Ask Chapter Name
    chapter_name = st.text_input("🔎 Enter Chapter or Lesson Name")

    if chapter_name:
        matched_docs = retriever.get_relevant_documents(chapter_name)
        if not matched_docs:
            st.error("❌ This chapter is not in the uploaded textbook.")
        else:
            st.success("✅ Chapter found.")

            task = st.radio("📌 Select a task", [
                "Extract Important Points",
                "Generate Summary",
                "Generate MCQs",
                "Generate Descriptive Questions and Answers",
                "Generate True/False Questions",
                "Generate Fill in the Blanks"
            ])

            if st.button("🧠 Generate Output"):
                prompts = {
                    "Extract Important Points": f"Extract all important points from the chapter: {chapter_name}",
                    "Generate Summary": f"Generate a detailed summary for the chapter: {chapter_name}",
                    "Generate MCQs": f"Generate 20 to 30 multiple choice questions with 4 options and answers from the chapter: {chapter_name}",
                    "Generate Descriptive Questions and Answers": f"Generate 8 to 10 descriptive questions and their detailed answers from the chapter: {chapter_name}",
                    "Generate True/False Questions": f"Generate 20 to 30 true/false questions based on the chapter: {chapter_name}",
                    "Generate Fill in the Blanks": f"Generate 20 to 30 fill-in-the-blank questions based on the chapter: {chapter_name}"
                }

                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff"
                )

                response = chain.run(prompts[task])
                st.text_area("📄 Generated Output", response, height=500)

                if st.button("📥 Download as PDF"):
                    buffer = BytesIO()
                    pdf = canvas.Canvas(buffer, pagesize=A4)
                    width, height = A4
                    text_obj = pdf.beginText(40, height - 50)
                    text_obj.setFont("Times-Roman", 12)

                    # Wrap and print text
                    for line in textwrap.wrap(response, width=100):
                        if text_obj.getY() < 50:
                            pdf.drawText(text_obj)
                            pdf.showPage()
                            text_obj = pdf.beginText(40, height - 50)
                            text_obj.setFont("Times-Roman", 12)
                        text_obj.textLine(line)

                    pdf.drawText(text_obj)
                    pdf.save()

                    st.download_button(
                        label="⬇️ Download Output as PDF",
                        data=buffer.getvalue(),
                        file_name=f"{chapter_name.replace(' ', '_')}_{task.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

    # Clean up temp file
    os.remove(tmp_path)