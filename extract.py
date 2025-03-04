import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import PyPDF2
import docx
import base64
from PIL import Image
import argparse
# import pytesseract
# import io

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def process_attachment(file_path, mime_type):
    try:
        if mime_type.startswith('image/'):
            return extract_from_image(file_path)
        elif mime_type == 'application/pdf':
            return extract_from_pdf(file_path)
        elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return extract_from_doc(file_path)
        elif mime_type == 'text/csv':
            return extract_from_csv(file_path)
        else:
            return f"Unsupported file type: {mime_type}"
    except Exception as e:
        return f"Error processing attachment: {str(e)}"

def extract_from_image(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        with Image.open(file_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode

        image_details = f"""
        Image Properties:
        - Dimensions: {width}x{height} pixels
        - Format: {format}
        - Color Mode: {mode}
        - File Size: {os.path.getsize(file_path) / 1024:.2f} KB
        """

        client = openai.OpenAI()  

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "user",
                "content": """
                Analyze the uploaded image and extract **all important details** present in it. Focus on identifying and organizing the following key elements:

                1. **Document Type & Purpose**  
                    - Determine whether the image is a receipt, invoice, report, form, or another type of document.  
                    - Identify its primary purpose (e.g., proof of purchase, financial record, information display).  

                2. **Key Data Points**  
                    - Extract and highlight all important numerical values, including:  
                    - Transaction amount, taxes, discounts, and total cost (if applicable).  
                    - Dates, times, and reference numbers.  
                    - Quantities and unit prices of listed items.  
                    - Provide context for the significance of these numbers.  

                3. **Visible Text Content**  
                    - Extract all readable text, including headings, labels, and descriptions.  
                    - Identify key sections such as store names, transaction details, and payment information.  
                    - Highlight any bold or emphasized text that may indicate importance.  

                4. **Entities & Identifiable Elements**  
                    - Recognize names, organizations, brands, store names, and locations.  
                    - Identify relationships between different entities, such as a business and customer.  

                5. **Payment & Transaction Information**  
                    - Identify payment methods (cash, card, digital payment).  
                    - Extract masked credit card numbers, approval codes, or bank details.  

                6. **Additional Notes or Instructions**  
                    - Extract any terms, policies, return instructions, or disclaimers.  
                    - Identify any calls to action.  

                **Output Format:**  
                - Organize the extracted details in a structured format with clear sections and bullet points.  
                - Maintain accuracy while preserving contextual relationships between extracted elements.  
                """
                }

            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content  

    except Exception as e:
        return f"Error processing image: {str(e)}"


def extract_from_pdf(file_path):
    try:
        text_content = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Please analyze this PDF content and extract all key information:
                    
                    1. Main topics and key points
                    2. Important facts, statistics, and data
                    3. Critical dates and deadlines
                    4. Names of significant people, organizations, and locations
                    5. Primary conclusions or findings
                    6. Action items or recommendations
                    7. Executive summary (if present)
                    8. Any tables, charts, or visual data with their significance
                    
                    Present the extracted information in an organized, easy-to-scan format. Highlight particularly crucial details or time-sensitive information.
                    
                    PDF Content:\n{text_content}
                    """
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def extract_from_doc(file_path):
    try:
        doc = docx.Document(file_path)
        text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Please analyze this document content and extract all key information:
                    
                    1. Core subject matter and main themes
                    2. Critical facts, figures, and statistics
                    3. Important dates, deadlines, and timeframes
                    4. Key people, organizations, and locations mentioned
                    5. Essential findings, conclusions, or outcomes
                    6. Required actions, next steps, or recommendations
                    7. Notable quotes or statements
                    8. Significant relationships or patterns between information
                    
                    Format the extracted information in a clear, structured manner. Prioritize the most crucial elements and indicate any time-sensitive information.
                    
                    Document Content:\n{text_content}
                    """
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Error processing document: {str(e)}"

def extract_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        
        analysis = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_summary": df.describe().to_dict(),
            "sample_data": df.head(5).to_dict(orient='records')
        }
        
        return {
            "statistical_summary": analysis,
            "preview": df.head(5).to_dict(orient='records')
        }

    except Exception as e:
        return f"Error processing CSV: {str(e)}"



def main():
    parser = argparse.ArgumentParser(description="Process and analyze files.")
    parser.add_argument("file_path", type=str, help="Path to the file to be processed")
    parser.add_argument("mime_type", type=str, help="MIME type of the file")
    
    args = parser.parse_args()
    
    result = process_attachment(args.file_path, args.mime_type)
    
    print("Processing Result:")
    print(result)

if __name__ == "__main__":
    main()