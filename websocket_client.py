import asyncio
import websockets
import json
import signal
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from datetime import datetime

console = Console()

class GmailWebSocketClient:
    def __init__(self, uri="ws://localhost:9000"):
        self.uri = uri
        self.websocket = None
        self.running = True
        self.in_progress_threads = {}
        
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        console.print("\nShutting down client...", style="bold red")
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())

    async def keepalive(self, websocket):
        while self.running:
            try:
                await websocket.send(json.dumps({"type": "keepalive"}))
                console.print("Sent keepalive message", style="dim")
                await asyncio.sleep(30)
            except websockets.exceptions.ConnectionClosedError:
                console.print("Connection closed while sending keepalive", style="bold red")
                break
            except Exception as e:
                console.print(f"Error in keepalive: {e}", style="bold red")
                break

    def display_sop_analysis(self, analysis_data):
        thread_id = analysis_data.get('thread_id')
        
        if thread_id in self.in_progress_threads:
            del self.in_progress_threads[thread_id]
            
        if "status" in analysis_data:
            if analysis_data["status"] == "in_progress":
                self.in_progress_threads[thread_id] = {
                    "started_at": datetime.now(),
                    "message": analysis_data.get("message", "Processing...")
                }
                console.print(Panel(
                    f"Thread ID: {thread_id}\n{analysis_data['message']}",
                    title="SOP Analysis Started",
                    style="bold blue"
                ))
                return
            elif analysis_data["status"] == "error":
                console.print(Panel(
                    f"Thread ID: {thread_id}\n{analysis_data['message']}",
                    title="SOP Analysis Error",
                    style="bold red"
                ))
                return
        
        if "skipped" in analysis_data and analysis_data["skipped"]:
            console.print(Panel(
                f"Thread ID: {thread_id}\n{analysis_data['message']}",
                title="Thread Skipped",
                style="bold yellow"
            ))
            return
        
        analysis = analysis_data["analysis"]
        if not analysis["analysis"]["needs_sop"]:
            console.print(Panel(
                f"Thread ID: {analysis['thread_id']}\nNo SOP needed for this conversation.",
                title="SOP Analysis Result",
                style="bold green"
            ))
            return

        department = analysis["analysis"]["department"]
        console.print(Panel(
            f"Department: {department}",
            title="Department Classification",
            style="bold blue"
        ))
        
        # Display relevant SOP topics if available
        if "relevant_topics" in analysis["analysis"] and analysis["analysis"]["relevant_topics"]:
            console.print(Panel(
                "\n".join([
                    "Related to the following SOP topics:",
                    *[f"- {topic}" for topic in analysis["analysis"]["relevant_topics"]]
                ]),
                title="Relevant SOP Topics",
                style="bold cyan"
            ))

        if not analysis["analysis"]["has_sufficient_info"]:
            console.print(Panel(
                "\n".join([
                    "SOP is needed, but more information is required.",
                    "\nSuggested questions to gather information:",
                    *[f"- {q}" for q in analysis["analysis"]["suggested_questions"]]
                ]),
                title="Information Needed",
                style="bold yellow"
            ))
        else:
            output_lines = [
                f"Title: {analysis['analysis']['sop_title']}",
                f"\nDescription: {analysis['analysis']['sop_description']}",
            ]
            
            if 'attachment_sources' in analysis['analysis'] and analysis['analysis']['attachment_sources']:
                output_lines.append("\nAttachment Sources:")
                for attachment in analysis['analysis']['attachment_sources']:
                    output_lines.append(f"- {attachment}")
            
            cleaned_steps = [
                step.lstrip('0123456789. ') 
                for step in analysis['analysis']['sop_steps']
            ]
            
            output_lines.extend([
                "\nSteps:",
                *[f"{i+1}. {step}" for i, step in enumerate(cleaned_steps)],
                f"\nPDF Generated: {analysis.get('pdf_path', 'N/A')}"
            ])
            
            console.print(Panel(
                "\n".join(output_lines),
                title="SOP Details",
                style="bold blue"
            ))
    
    def display_in_progress_threads(self):
        """Display a table of all currently in-progress threads"""
        if not self.in_progress_threads:
            return
            
        now = datetime.now()
        table = Table(title="In-Progress SOP Analyses")
        table.add_column("Thread ID", style="cyan")
        table.add_column("Time Elapsed", style="green")
        table.add_column("Status", style="yellow")
        
        for thread_id, info in self.in_progress_threads.items():
            elapsed = now - info["started_at"]
            elapsed_str = f"{int(elapsed.total_seconds())}s"
            table.add_row(thread_id, elapsed_str, info["message"])
            
        console.print(table)
        
    async def receive_messages(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                console.print(f"Connected to {self.uri}", style="bold green")
                console.print("Waiting for updates...", style="bold blue")
                
                keepalive_task = asyncio.create_task(self.keepalive(websocket))
                
                status_display_task = asyncio.create_task(self.periodic_status_display())
                
                while self.running:
                    try:
                        message = await websocket.recv()
                        
                        try:
                            data = json.loads(message)
                            
                            if data.get("type") == "email_update":
                                console.print("\nNew Email Update:", style="bold green")
                                console.print(json.dumps(data["data"], indent=2))
                                
                            elif data.get("type") == "attachment_status":
                                console.print("\nAttachment Status Update:", style="bold yellow")
                                console.print(json.dumps(data["data"], indent=2))
                                
                            elif data.get("type") == "sop_analysis":
                                console.print("\nSOP Analysis Result:", style="bold magenta")
                                self.display_sop_analysis(data["data"])
                                
                            elif data.get("type") == "keepalive_ack":
                                console.print("Received keepalive acknowledgment", style="dim")
                                
                        except json.JSONDecodeError:
                            console.print(f"Received invalid JSON: {message}", style="yellow")
                            
                    except websockets.exceptions.ConnectionClosedError as e:
                        console.print(f"\nConnection lost with error: {e}", style="bold red")
                        break
                    except Exception as e:
                        console.print(f"Error receiving message: {e}", style="bold red")
                        import traceback
                        traceback.print_exc()
                        if self.running:
                            await asyncio.sleep(1)
                            continue
                        else:
                            break
                
                keepalive_task.cancel()
                status_display_task.cancel()
                try:
                    await keepalive_task
                    await status_display_task
                except asyncio.CancelledError:
                    pass
                    
        except websockets.exceptions.ConnectionClosedError as e:
            console.print(f"Connection error: {e}", style="bold red")
        except Exception as e:
            console.print(f"Error: {e}", style="bold red")
            import traceback
            traceback.print_exc()
        finally:
            self.websocket = None
    
    async def periodic_status_display(self):
        """Periodically update the display of in-progress threads"""
        while self.running:
            try:
                if self.in_progress_threads:
                    self.display_in_progress_threads()
                await asyncio.sleep(5) 
            except Exception as e:
                console.print(f"Error updating status display: {e}", style="dim red")
                await asyncio.sleep(5)
            
    async def run(self):
        while self.running:
            try:
                await self.receive_messages()
                if self.running:
                    console.print("Reconnecting in 5 seconds...", style="bold yellow")
                    await asyncio.sleep(5)
            except Exception as e:
                console.print(f"Error: {e}", style="bold red")
                if self.running:
                    await asyncio.sleep(5)

if __name__ == "__main__":
    client = GmailWebSocketClient()
    asyncio.run(client.run())