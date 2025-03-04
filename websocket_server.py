import asyncio
import json
import websockets
from datetime import datetime
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time
from fetch_gmail import get_gmail_service, check_new_emails
from sop_analyzer import EmailConversationAnalyzer

attachment_status_queue = Queue()
sop_analysis_queue = Queue()
thread_processing_queue = Queue()

class GmailWebSocketServer:
    def __init__(self, host="localhost", port=9000):
        self.host = host
        self.port = port
        self.gmail_service = get_gmail_service()
        self.connected_clients = set()
        self.check_interval = 10
        self.sop_analyzer = EmailConversationAnalyzer()
        self.threads_in_process = set()
        self.max_concurrent_tasks = 3
        self.analysis_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.task_processor = None
    
    def parse_email_data(self, email_data):
        try:
            if isinstance(email_data, str):
                try:
                    return json.loads(email_data)
                except json.JSONDecodeError:
                    return {"message": email_data}
            elif isinstance(email_data, (dict, list)):
                return email_data
            else:
                return {"message": str(email_data)}
        except Exception as e:
            return {"message": f"Error processing email data: {str(e)}"}
    
    async def analyze_sop_for_thread(self, thread_data):
        thread_id = thread_data["thread_id"]
        
        if thread_id in self.threads_in_process:
            print(f"Thread {thread_id} is already being processed, skipping duplicate analysis")
            return None
            
        self.threads_in_process.add(thread_id)
        
        try:
            async with self.analysis_semaphore:
                print(f"Starting SOP analysis for thread: {thread_id}")
                conversation_data = {
                    "thread_id": thread_id,
                    "subject": thread_data["subject"],
                    "conversation": []
                }
                
                for msg in thread_data["conversation"]:
                    message_info = {
                        "timestamp": msg["timestamp"],
                        "sender": msg["sender"],
                        "body": msg["body"]
                    }
                    
                    if "attachments" in msg and msg["attachments"]:
                        message_info["attachments"] = msg["attachments"]
                    
                    conversation_data["conversation"].append(message_info)
                
                status_update = {
                    "thread_id": thread_id,
                    "status": "in_progress",
                    "message": f"SOP analysis started for thread: {thread_id}"
                }
                sop_analysis_queue.put(status_update)
                
                analysis_result = await self.sop_analyzer.analyze_conversation(conversation_data)
                
                if "skipped" in analysis_result and analysis_result["skipped"]:
                    print(f"Skipped thread {thread_id}: {analysis_result['message']}")
                    sop_analysis_queue.put({
                        "thread_id": thread_id,
                        "skipped": True,
                        "message": analysis_result["message"]
                    })
                else:
                    sop_analysis_queue.put({
                        "thread_id": thread_id,
                        "analysis": analysis_result
                    })
                
                return analysis_result
                
        except Exception as e:
            print(f"Error analyzing SOP for thread {thread_id}: {e}")
            import traceback
            traceback.print_exc()
            
            sop_analysis_queue.put({
                "thread_id": thread_id,
                "status": "error",
                "message": f"Error analyzing thread: {str(e)}"
            })
            
            return None
        finally:

            if thread_id in self.threads_in_process:
                self.threads_in_process.remove(thread_id)

    async def process_thread_queue(self):
        """Background task to process the thread queue"""
        while True:
            try:
                if not thread_processing_queue.empty():
                    thread_data = thread_processing_queue.get_nowait()
                    asyncio.create_task(self.analyze_sop_for_thread(thread_data))
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in thread queue processing: {e}")
                await asyncio.sleep(1)

    async def handle_client(self, websocket):
        attachment_monitor_task = None
        sop_monitor_task = None
        try:
            self.connected_clients.add(websocket)
            print(f"New client connected. Total clients: {len(self.connected_clients)}")
            
            initial_data = check_new_emails(self.gmail_service)
            parsed_data = self.parse_email_data(initial_data)
            
            await websocket.send(json.dumps({
                "type": "email_update",
                "data": parsed_data
            }))
            
            attachment_monitor_task = asyncio.create_task(
                self.monitor_attachment_status(websocket)
            )
            sop_monitor_task = asyncio.create_task(
                self.monitor_sop_analysis(websocket)
            )
            
            if isinstance(parsed_data, list):
                for thread in parsed_data:
                    thread_processing_queue.put(thread)
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=60)
                    try:
                        msg_data = json.loads(message)
                        if msg_data.get("type") == "keepalive":
                            print("Received keepalive from client")
                            await websocket.send(json.dumps({"type": "keepalive_ack"}))
                            continue
                        else:
                            print(f"Received message from client: {msg_data}")
                    except json.JSONDecodeError:
                        print(f"Received invalid JSON from client: {message}")
                except asyncio.TimeoutError:
                    try:
                        pong_waiter = await websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                        print("Ping successful, connection still active")
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                        print("Ping failed, client disconnected")
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Client disconnected")
                    break
                    
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Client connection closed with error: {e}")
        except Exception as e:
            print(f"Error handling client: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
            print(f"Client disconnected. Remaining clients: {len(self.connected_clients)}")
            if attachment_monitor_task:
                attachment_monitor_task.cancel()
            if sop_monitor_task:
                sop_monitor_task.cancel()

    async def monitor_attachment_status(self, websocket):
        while True:
            try:
                while not attachment_status_queue.empty():
                    status = attachment_status_queue.get_nowait()
                    await websocket.send(json.dumps({
                        "type": "attachment_status",
                        "data": status
                    }))
                await asyncio.sleep(1)
            except websockets.exceptions.ConnectionClosedError:
                break
            except Exception as e:
                print(f"Error in attachment monitoring: {e}")
                await asyncio.sleep(1)

    async def monitor_sop_analysis(self, websocket):
        while True:
            try:
                while not sop_analysis_queue.empty():
                    analysis = sop_analysis_queue.get_nowait()
                    await websocket.send(json.dumps({
                        "type": "sop_analysis",
                        "data": analysis
                    }))
                await asyncio.sleep(1)
            except websockets.exceptions.ConnectionClosedError:
                break
            except Exception as e:
                print(f"Error in SOP analysis monitoring: {e}")
                await asyncio.sleep(1)

    async def check_emails_periodically(self):
        while True:
            try:
                if not self.connected_clients:
                    await asyncio.sleep(self.check_interval)
                    continue
                    
                email_data = check_new_emails(self.gmail_service)
                parsed_data = self.parse_email_data(email_data)
                
                if parsed_data and not (
                    isinstance(parsed_data, dict) and 
                    parsed_data.get("message", "").startswith("No new messages")
                ):
                    email_message = json.dumps({
                        "type": "email_update",
                        "data": parsed_data
                    })
                    
                    for client in list(self.connected_clients):
                        try:
                            await client.send(email_message)
                        except websockets.exceptions.ConnectionClosedError:
                            self.connected_clients.remove(client)
                        except Exception as e:
                            print(f"Error sending to client: {e}")
                    
                    if isinstance(parsed_data, list):
                        for thread in parsed_data:
                            thread_processing_queue.put(thread)
                
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"Error checking emails: {e}")
                await asyncio.sleep(self.check_interval)

    def run(self):
        async def main():
            self.task_processor = asyncio.create_task(self.process_thread_queue())
            
            async with websockets.serve(
                self.handle_client, 
                self.host, 
                self.port
            ):
                print(f"Server is running on ws://{self.host}:{self.port}")
                print(f"Processing up to {self.max_concurrent_tasks} threads simultaneously")
                await self.check_emails_periodically()

        print("Starting WebSocket server...")
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            if self.task_processor:
                self.task_processor.cancel()

if __name__ == "__main__":
    server = GmailWebSocketServer()
    server.run()