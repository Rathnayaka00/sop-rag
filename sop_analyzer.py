import os
import json
import openai
from typing import Dict, List, Optional, Set
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

console = Console()

class EmailConversationAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.console = Console()
        self.processed_threads = set()
        self.load_processed_threads()
        
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./pdf_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        
        self.sop_topics = self.load_sop_topics()
        
    def load_sop_topics(self) -> List[str]:
        try:
            if os.path.exists('sop_topics.json'):
                with open('sop_topics.json', 'r') as f:
                    data = json.load(f)
                    return data.get('sop_topics', [])
            else:
                console.print("[bold yellow]Warning: sop_topics.json not found. Using empty topics list.[/bold yellow]")
                return []
        except Exception as e:
            console.print(f"[bold red]Error loading SOP topics: {e}[/bold red]")
            return []
        
    def load_processed_threads(self) -> None:
        try:
            if os.path.exists('processed_threads.json'):
                with open('processed_threads.json', 'r') as f:
                    self.processed_threads = set(json.load(f))
                console.print(f"Loaded {len(self.processed_threads)} previously processed threads")
        except Exception as e:
            console.print(f"[bold red]Error loading processed threads: {e}[/bold red]")
            self.processed_threads = set()
            
    def save_processed_threads(self) -> None:
        try:
            with open('processed_threads.json', 'w') as f:
                json.dump(list(self.processed_threads), f)
        except Exception as e:
            console.print(f"[bold red]Error saving processed threads: {e}[/bold red]")
        
    def extract_clean_conversation(self, conversation: List[Dict]) -> List[Dict]:
        clean_messages = []
        for message in conversation:
            body = message['body'].split('\r\n\r\nOn')[0].strip()
            sender_name = message['sender']
            if '<' in sender_name:
                sender_name = sender_name.split('<')[0].strip()
            
            message_data = {
                'timestamp': message['timestamp'],
                'sender': sender_name,
                'message': body,
            }
            
            if 'attachments' in message and message['attachments']:
                attachments_data = []
                for attachment in message['attachments']:
                    attachment_info = {
                        'filename': attachment['filename'],
                        'type': attachment['type'],
                        'size': attachment['size']
                    }
                    
                    if 'extracted_data' in attachment:
                        attachment_info['content'] = attachment['extracted_data']
                    
                    attachments_data.append(attachment_info)
                
                message_data['attachments'] = attachments_data
            
            clean_messages.append(message_data)
        return clean_messages

    def is_conversation_concluded(self, conversation: List[Dict]) -> bool:
        if not conversation:
            return True

        last_sender = conversation[-1]['sender'].lower()
        if "malinda rathnayaka" in last_sender:
            return True
            
        last_message = conversation[-1]['message'].lower()
        concluding_phrases = [
            'thank', 'thanks', 'ok', 'will do', 'understood', 
            'approved', 'confirmed', 'noted', 'acknowledged'
        ]
        
        return any(phrase in last_message for phrase in concluding_phrases)

    def identify_conversation_topics(self, subject: str, conversation_text: str) -> List[str]:
        prompt = f"""
        Extract the 3-5 main topics or keywords from this email conversation:
        
        Subject: {subject}
        
        Conversation:
        {conversation_text}
        
        Return ONLY a list of keywords or phrases that represent the main topics, one per line.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        topics = response.choices[0].message.content.strip().split("\n")
        return [topic.strip("- ").lower() for topic in topics]
    
    def is_sop_relevant(self, conversation_topics: List[str]) -> bool:
        if not self.sop_topics:
            return False
            
        sop_topics_lower = [topic.lower() for topic in self.sop_topics]
        
        for topic in conversation_topics:
            for sop_topic in sop_topics_lower:
                if topic in sop_topic or sop_topic in topic:
                    console.print(f"[bold green]Matched topic: {topic} with SOP topic: {sop_topic}[/bold green]")
                    return True
                    
        return False
        
    def retrieve_relevant_sop_content(self, query: str, num_results: int = 5) -> str:
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results
            )
            
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]
                return "\n".join(documents)
            else:
                return ""
        except Exception as e:
            console.print(f"[bold red]Error retrieving SOP content: {e}[/bold red]")
            return ""

    def generate_sop_pdf(self, analysis_result: Dict) -> str:
        filename = f"SOP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        ))
        
        styles.add(ParagraphStyle(
            name='Step',
            parent=styles['Normal'],
            leftIndent=20,
            spaceBefore=6,
            spaceAfter=6
        ))
        
        content = []
        
        department = analysis_result['analysis'].get('department', 'Unspecified Department')
        content.append(Paragraph(f"Department: {department}", styles['CustomHeading']))
        content.append(Spacer(1, 12))
        
        content.append(Paragraph(analysis_result['analysis']['sop_title'], styles['Title']))
        content.append(Spacer(1, 12))
        
        content.append(Paragraph(f"Email Subject: {analysis_result['subject']}", styles['Normal']))
        content.append(Spacer(1, 12))
        
        if 'attachment_sources' in analysis_result['analysis'] and analysis_result['analysis']['attachment_sources']:
            content.append(Paragraph("Referenced Attachments:", styles['Heading2']))
            for attachment in analysis_result['analysis']['attachment_sources']:
                content.append(Paragraph(f"• {attachment}", styles['Normal']))
            content.append(Spacer(1, 12))
        
        content.append(Paragraph("Description:", styles['Heading2']))
        content.append(Paragraph(analysis_result['analysis']['sop_description'], styles['Normal']))
        content.append(Spacer(1, 12))
        
        content.append(Paragraph("Procedure Steps:", styles['Heading2']))
        for i, step in enumerate(analysis_result['analysis']['sop_steps'], 1):
            step_text = step.lstrip('0123456789. ')
            content.append(Paragraph(f"{i}. {step_text}", styles['Step']))
        
        content.append(Spacer(1, 20))
        content.append(Paragraph(
            f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        
        doc.build(content)
        return filename

    async def analyze_conversation(self, thread_data: Dict) -> Dict:
        thread_id = thread_data["thread_id"]
        
        if thread_id in self.processed_threads:
            return {
                "thread_id": thread_id,
                "skipped": True,
                "message": "Thread already processed. SOP previously generated."
            }
        
        with console.status("[bold green]Analyzing conversation and attachments...") as status:
            clean_conversation = self.extract_clean_conversation(thread_data['conversation'])

            if clean_conversation and "malinda rathnayaka" in clean_conversation[-1]['sender'].lower():
                return {
                    "thread_id": thread_id,
                    "skipped": True,
                    "message": "Analysis skipped - last message from Malinda Rathnayaka."
                }
        
            is_concluded = self.is_conversation_concluded(clean_conversation)
            
            conversation_parts = []
            for msg in clean_conversation:
                msg_text = f"{msg['sender']} ({msg['timestamp']}): {msg['message']}"
                
                if 'attachments' in msg and msg['attachments']:
                    msg_text += "\n--- Attachments ---"
                    for i, attachment in enumerate(msg['attachments'], 1):
                        msg_text += f"\n[Attachment {i}] {attachment['filename']} ({attachment['type']}, {attachment['size']})"
                        if 'content' in attachment:
                            msg_text += f"\nExtracted content:\n{attachment['content']}"
                
                conversation_parts.append(msg_text)
            
            conversation_text = "\n\n".join(conversation_parts)
            
            conversation_topics = self.identify_conversation_topics(thread_data['subject'], conversation_text)
            console.print(f"[bold blue]Identified conversation topics: {', '.join(conversation_topics)}[/bold blue]")
            
            is_sop_relevant = self.is_sop_relevant(conversation_topics)
            
            if is_sop_relevant:
                console.print("[bold green]Conversation is relevant to existing SOPs. Using RAG for enhanced analysis.[/bold green]")
                
                query = f"Subject: {thread_data['subject']}\nTopics: {', '.join(conversation_topics)}"
                relevant_sop_content = self.retrieve_relevant_sop_content(query)
                
                functions = [{
                    "name": "analyze_conversation_needs",
                    "description": "Analyze if conversation needs SOP and generate appropriate response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "needs_sop": {
                                "type": "boolean",
                                "description": "Whether the conversation requires an SOP"
                            },
                            "conversation_state": {
                                "type": "string",
                                "enum": ["ongoing", "concluded"],
                                "description": "Current state of the conversation"
                            },
                            "has_sufficient_info": {
                                "type": "boolean",
                                "description": "Whether there's enough information to generate SOP"
                            },
                            "department": {
                                "type": "string",
                                "description": "Main department category (e.g., 'Human Resources (HR) & Administration', 'Finance & Accounting', 'IT & Security')"
                            },
                            "attachment_sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of attachment filenames that contributed to SOP creation"
                            },
                            "suggested_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions to gather more information if needed"
                            },
                            "sop_title": {
                                "type": "string",
                                "description": "Title for the SOP if needed"
                            },
                            "sop_description": {
                                "type": "string",
                                "description": "Brief description of the SOP purpose"
                            },
                            "sop_steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Detailed steps for the SOP"
                            }
                        },
                        "required": ["needs_sop", "conversation_state", "department"]
                    }
                }]
                
                prompt = f"""
                Analyze this email conversation comprehensively, including any attachments:
                
                Subject: {thread_data['subject']}
                
                Conversation:
                {conversation_text}
                
                RELEVANT SOP INFORMATION:
                {relevant_sop_content}
                
                Tasks:
                1. Determine if this conversation indicates a process that needs an SOP
                2. Assess if the conversation is ongoing or concluded
                3. Identify the main department category this conversation belongs to
                4. If SOP is needed:
                - Check if there's sufficient information to create the SOP
                - If information is insufficient, suggest specific questions to gather more details
                - If information is sufficient, create detailed SOP steps
                - Incorporate relevant information from attachments (if any)
                - List which attachments contributed to the SOP
                - Incorporate the retrieved SOP information where relevant
                """

                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    functions=functions,
                    function_call={"name": "analyze_conversation_needs"}
                )

                analysis = json.loads(response.choices[0].message.function_call.arguments)
                
                analysis["used_rag"] = True
                
            else:
                console.print("[bold yellow]Conversation not directly related to existing SOPs. Using standard analysis.[/bold yellow]")
                
                functions = [{
                    "name": "analyze_conversation_needs",
                    "description": "Analyze if conversation needs SOP and generate appropriate response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "needs_sop": {
                                "type": "boolean",
                                "description": "Whether the conversation requires an SOP"
                            },
                            "conversation_state": {
                                "type": "string",
                                "enum": ["ongoing", "concluded"],
                                "description": "Current state of the conversation"
                            },
                            "has_sufficient_info": {
                                "type": "boolean",
                                "description": "Whether there's enough information to generate SOP"
                            },
                            "department": {
                                "type": "string",
                                "description": "Main department category (e.g., 'Human Resources (HR) & Administration', 'Finance & Accounting', 'IT & Security')"
                            },
                            "attachment_sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of attachment filenames that contributed to SOP creation"
                            },
                            "suggested_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions to gather more information if needed"
                            },
                            "sop_title": {
                                "type": "string",
                                "description": "Title for the SOP if needed"
                            },
                            "sop_description": {
                                "type": "string",
                                "description": "Brief description of the SOP purpose"
                            },
                            "sop_steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Detailed steps for the SOP"
                            }
                        },
                        "required": ["needs_sop", "conversation_state", "department"]
                    }
                }]
                
                prompt = f"""
                Analyze this email conversation comprehensively, including any attachments:
                
                Subject: {thread_data['subject']}
                
                Conversation:
                {conversation_text}
                
                Tasks:
                1. Determine if this conversation indicates a process that needs an SOP
                2. Assess if the conversation is ongoing or concluded
                3. Identify the main department category this conversation belongs to
                4. If SOP is needed:
                - Check if there's sufficient information to create the SOP
                - If information is insufficient, suggest specific questions to gather more details
                - If information is sufficient, create detailed SOP steps
                - Incorporate relevant information from attachments (if any)
                - List which attachments contributed to the SOP
                """

                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    functions=functions,
                    function_call={"name": "analyze_conversation_needs"}
                )

                analysis = json.loads(response.choices[0].message.function_call.arguments)
                
                analysis["used_rag"] = False
            
            result = {
                "thread_id": thread_id,
                "subject": thread_data["subject"],
                "is_concluded": is_concluded,
                "analysis": analysis
            }
            
            if analysis['needs_sop'] and analysis['has_sufficient_info']:
                pdf_path = self.generate_sop_pdf(result)
                result['pdf_path'] = pdf_path
                
                self.processed_threads.add(thread_id)
                self.save_processed_threads()
            
            return result

    def format_output(self, analysis_result: Dict) -> None:
        if analysis_result.get('skipped', False):
            console.print(Panel(
                f"Thread ID: {analysis_result['thread_id']}\n{analysis_result['message']}",
                title="Thread Skipped",
                style="bold yellow"
            ))
            return

        if not analysis_result['analysis']['needs_sop']:
            console.print(Panel(
                f"Thread ID: {analysis_result['thread_id']}\nNo SOP needed for this conversation.",
                title="Analysis Result",
                style="bold green"
            ))
            return

        if analysis_result['analysis']['needs_sop']:
            department = analysis_result['analysis']['department']
            used_rag = analysis_result['analysis'].get('used_rag', False)
            rag_status = "[RAG Enhanced]" if used_rag else "[Standard Analysis]"
            
            console.print(Panel(
                f"Department: {department} {rag_status}",
                title="Department Classification",
                style="bold blue"
            ))

            if not analysis_result['analysis']['has_sufficient_info']:
                console.print(Panel(
                    "\n".join([
                        "SOP is needed, but more information is required.",
                        "\nSuggested questions to gather information:",
                        *[f"- {q}" for q in analysis_result['analysis']['suggested_questions']]
                    ]),
                    title="Information Needed",
                    style="bold yellow"
                ))
            else:
                output_lines = [
                    f"Title: {analysis_result['analysis']['sop_title']}",
                    f"\nDescription: {analysis_result['analysis']['sop_description']}",
                ]
                
                if 'attachment_sources' in analysis_result['analysis'] and analysis_result['analysis']['attachment_sources']:
                    output_lines.append("\nAttachment Sources:")
                    for attachment in analysis_result['analysis']['attachment_sources']:
                        output_lines.append(f"- {attachment}")
                
                cleaned_steps = [
                    step.lstrip('0123456789. ') 
                    for step in analysis_result['analysis']['sop_steps']
                ]
                
                output_lines.extend([
                    "\nSteps:",
                    *[f"{i+1}. {step}" for i, step in enumerate(cleaned_steps)],
                    f"\nPDF Generated: {analysis_result.get('pdf_path', 'N/A')}"
                ])
                
                console.print(Panel(
                    "\n".join(output_lines),
                    title=f"SOP Details {rag_status}",
                    style="bold blue"
                ))