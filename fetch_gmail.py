import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import base64
import json
import pickle
from email.mime.text import MIMEText
import shelve
from datetime import datetime, timedelta
from collections import defaultdict
import mimetypes
import io
import sys
import hashlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import extract

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
STORAGE_FILE = 'email_storage'
ATTACHMENT_DIR = 'attachments'
ATTACHMENT_CACHE_FILE = 'attachment_cache'

class AttachmentCache:
    def __init__(self):
        self.cache_file = ATTACHMENT_CACHE_FILE
        
    def _generate_hash(self, file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def get(self, file_path):
        if not os.path.exists(file_path):
            return None
        file_hash = self._generate_hash(file_path)
        with shelve.open(self.cache_file) as cache:
            return cache.get(file_hash)
            
    def set(self, file_path, extracted_data):
        file_hash = self._generate_hash(file_path)
        with shelve.open(self.cache_file) as cache:
            cache[file_hash] = extracted_data

def get_gmail_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def get_stored_threads():
    with shelve.open(STORAGE_FILE) as storage:
        return dict(storage.get('processed_threads', {}))

def save_threads(threads):
    with shelve.open(STORAGE_FILE) as storage:
        storage['processed_threads'] = dict(threads)

def ensure_attachment_dir():
    if not os.path.exists(ATTACHMENT_DIR):
        os.makedirs(ATTACHMENT_DIR)

def download_attachment(service, message_id, attachment_id, filename):
    ensure_attachment_dir()
    attachment = service.users().messages().attachments().get(
        userId='me',
        messageId=message_id,
        id=attachment_id
    ).execute()
    file_data = base64.urlsafe_b64decode(attachment['data'])
    file_path = os.path.join(ATTACHMENT_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(file_data)
    return file_path

attachment_cache = AttachmentCache()

def get_attachment_info(service, message_id, part):
    if 'filename' not in part.get('body', {}) and 'filename' not in part:
        return None
    filename = part.get('filename', '')
    if not filename:
        return None
    
    mime_type = part.get('mimeType', 'application/octet-stream')
    size = part['body'].get('size', 0)
    attachment_id = part['body'].get('attachmentId', '')
    
    if attachment_id:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_filename = f"{message_id}_{timestamp}_{filename}"
        file_path = download_attachment(service, message_id, attachment_id, unique_filename)
        
        cached_data = attachment_cache.get(file_path)
        extracted_data = cached_data if cached_data is not None else extract.process_attachment(file_path, mime_type)
        
        if cached_data is None:
            attachment_cache.set(file_path, extracted_data)
        
        return {
            'filename': filename,
            'mime_type': mime_type,
            'size': size,
            'local_path': file_path,
            'extracted_data': extracted_data
        }
    
    return {
        'filename': filename,
        'mime_type': mime_type,
        'size': size,
        'attachment_id': attachment_id
    }

def get_email_content(service, message_id, process_attachments=False):
    try:
        message = service.users().messages().get(
            userId='me', 
            id=message_id, 
            format='full'
        ).execute()

        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown Sender")
        thread_id = message['threadId']

        body = ''
        attachments = []

        def process_parts(parts):
            nonlocal body
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                elif 'parts' in part:
                    process_parts(part['parts'])
                elif process_attachments:  
                    attachment = get_attachment_info(service, message_id, part)
                    if attachment:
                        attachments.append(attachment)

        if 'parts' in message['payload']:
            process_parts(message['payload']['parts'])
        else:
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']
                ).decode('utf-8')

        timestamp = datetime.fromtimestamp(
            int(message['internalDate'])/1000
        ).strftime('%Y-%m-%d %H:%M:%S')

        is_sent_by_me = any(
            h['value'] == 'me' 
            for h in headers 
            if h['name'].lower() == 'from'
        )

        return {
            'id': message_id,
            'thread_id': thread_id,
            'sender': sender,
            'subject': subject,
            'body': body,
            'timestamp': timestamp,
            'is_sent_by_me': is_sent_by_me,
            'attachments': attachments
        }

    except Exception as e:
        print(f"Error processing message {message_id}: {str(e)}")
        return None

def get_thread_messages(service, thread_id, processed_threads=None):
    try:
        thread = service.users().threads().get(
            userId='me', 
            id=thread_id
        ).execute()
        
        previous_messages = {}
        if processed_threads and thread_id in processed_threads:
            for msg in processed_threads[thread_id]['messages']:
                previous_messages[msg['id']] = msg
        
        messages = []
        for message in thread['messages']:
            message_id = message['id']
            
            if message_id in previous_messages:
                messages.append(previous_messages[message_id])
            else:
                email_data = get_email_content(service, message_id, process_attachments=True)
                if email_data:
                    messages.append(email_data)
        
        messages.sort(key=lambda x: x['timestamp'])
        return messages
    
    except Exception as e:
        print(f"Error getting thread {thread_id}: {str(e)}")
        return []

def check_new_emails(service, max_results=10):
    try:
        processed_threads = get_stored_threads()
        
        results = service.users().threads().list(
            userId='me', 
            labelIds=['INBOX'], 
            maxResults=max_results
        ).execute()

        threads = results.get('threads', [])
        if not threads:
            return "No email threads found in inbox."

        new_threads = defaultdict(list)
        updated_threads = {}
        
        for thread in threads:
            thread_id = thread['id']
            thread_messages = get_thread_messages(service, thread_id, processed_threads)
            
            if not thread_messages:
                continue
                
            if thread_id not in processed_threads:
                new_threads[thread_id] = {
                    'subject': thread_messages[0]['subject'],
                    'messages': thread_messages
                }
            else:
                existing_message_count = len(processed_threads[thread_id]['messages'])
                if len(thread_messages) > existing_message_count:
                    new_threads[thread_id] = {
                        'subject': thread_messages[0]['subject'],
                        'messages': thread_messages
                    }
            
            updated_threads[thread_id] = {
                'subject': thread_messages[0]['subject'],
                'messages': thread_messages
            }

        save_threads(updated_threads)

        if new_threads:
            output = []
            for thread_id, thread_data in new_threads.items():
                thread_info = {
                    'thread_id': thread_id,
                    'subject': thread_data['subject'],
                    'conversation': []
                }
                
                for msg in thread_data['messages']:
                    message_info = {
                        'timestamp': msg['timestamp'],
                        'sender': msg['sender'],
                        'body': msg['body'],
                        'attachments': []
                    }
                    
                    for att in msg.get('attachments', []):
                        attachment_info = {
                            'filename': att['filename'],
                            'type': att['mime_type'],
                            'size': f"{att.get('size', 0) / 1024:.1f} KB"
                        }
                        if 'extracted_data' in att:
                            attachment_info['extracted_data'] = att['extracted_data']
                        message_info['attachments'].append(attachment_info)
                    
                    thread_info['conversation'].append(message_info)
                
                output.append(thread_info)
            
            output.sort(
                key=lambda x: max(
                    (msg['timestamp'] for msg in x['conversation']),
                    default="1970-01-01 00:00:00"
                ),
                reverse=True
            )
            
            return json.dumps(output, indent=2)
        else:
            return f"No new messages in any threads. {len(processed_threads)} threads have been processed previously."

    except Exception as e:
        print(f"Error checking new emails: {str(e)}")
        return f"Error occurred while checking emails: {str(e)}"