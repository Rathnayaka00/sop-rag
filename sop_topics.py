import fitz             
import json
import openai  
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env file")

openai.api_key = OPENAI_API_KEY  

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_topics_from_text(text):
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract the key topics from the given company SOP document."},
            {"role": "user", "content": f"Extract the main topics from this text:\n{text}"}
        ]
    )
    
    topics = response.choices[0].message.content.strip().split("\n")  
    return [topic.strip("- ") for topic in topics]  

def store_topics_locally(topics, file_path="sop_topics.json"):

    with open(file_path, "w") as f:
        json.dump({"sop_topics": topics}, f, indent=4)
    print(f"Topics stored in {file_path}")


pdf_path = r"document/sop.pdf"  
sop_text = extract_text_from_pdf(pdf_path)
topics = extract_topics_from_text(sop_text)
store_topics_locally(topics)
