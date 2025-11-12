#!/usr/bin/env python3
"""
Simplified Socket Voice Server 
Ensures background RAG processing completes properly
"""

import socket
import threading
import json
import logging
import time
from voice_chatbot import VoiceChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedSocketServer:
    """Socket server that creates individual sessions per client."""
    
    def __init__(self, host='localhost', port=8889):
        self.host = host
        self.port = port
        self.clients = {}  # Store client sessions: {client_id: {'chatbot': VoiceChatbot, 'session_id': str}}
        self.client_counter = 0
        
        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def create_client_session(self, client_id: str) -> dict:
        """Create a new VoiceChatbot instance and session for a client."""
        print(f"🚀 Creating new session for client {client_id}")
        
        # Create individual VoiceChatbot instance
        chatbot = VoiceChatbot()
        session_id = chatbot.rag_orchestrator.create_session()
        chatbot.session_id = session_id
        
        client_data = {
            'chatbot': chatbot,
            'session_id': session_id,
            'created_at': time.time()
        }
        
        self.clients[client_id] = client_data
        print(f"✅ Session {session_id} created for client {client_id}")
        return client_data
    
    def cleanup_client_session(self, client_id: str):
        """Clean up client session and expire it."""
        if client_id in self.clients:
            client_data = self.clients[client_id]
            session_id = client_data['session_id']
            
            # You could add session cleanup here if needed
            # client_data['chatbot'].rag_orchestrator.expire_session(session_id)
            
            del self.clients[client_id]
            print(f"🗑️ Session {session_id} cleaned up for client {client_id}")
    
    def emit_to_client(self, client_socket, event_type: str, data: str):
        """Emit data to client."""
        try:
            message = {"type": event_type, "data": data}
            client_socket.send((json.dumps(message) + '\n').encode('utf-8'))
        except:
            pass
            
    def handle_client(self, client_socket, client_address):
        """Handle client connection with individual session."""
        self.client_counter += 1
        client_id = f"client_{self.client_counter}_{client_address[0]}_{client_address[1]}"
        
        print(f"🔗 Client {client_id} connected from {client_address}")
        
        # Create individual session for this client
        client_data = self.create_client_session(client_id)
        session_id = client_data['session_id']
        
        self.emit_to_client(client_socket, "connected", f"Connected! Your session ID: {session_id}")
        
        try:
            while True:
                data = client_socket.recv(1024).decode('utf-8').strip()
                if not data:
                    break
                    
                # Parse query
                try:
                    message = json.loads(data)
                    query = message.get('query', '').strip()
                except:
                    query = data.strip()
                    
                if query:
                    self.process_with_individual_session(client_socket, client_id, query)
                    
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
        finally:
            # Clean up session when client disconnects
            self.cleanup_client_session(client_id)
            client_socket.close()
            print(f"🔌 Client {client_id} disconnected - session expired")
            
    def process_with_individual_session(self, client_socket, client_id: str, query: str):
        """Process query using client's individual session."""
        try:
            if client_id not in self.clients:
                self.emit_to_client(client_socket, "error", "Session not found")
                return
                
            client_data = self.clients[client_id]
            chatbot = client_data['chatbot']
            session_id = client_data['session_id']
            
            print(f"\n🗣️ Client {client_id} | Session {session_id}: '{query}'")
            self.emit_to_client(client_socket, "status", f"Processing: {query}")
            
            # Capture all speaks in a list to emit them in order
            responses = []
            
            # Store original speak
            original_speak = chatbot.speak_text
            
            def capture_speak(text):
                responses.append(text)
                self.emit_to_client(client_socket, "response", text)
                print(f"🔊 Client {client_id}: {text[:50]}...")
            
            # Set the capture function
            chatbot.speak_text = capture_speak
            
            try:
                # Process the query with individual session
                print(f"🔄 Processing with session {session_id}...")
                success = chatbot.process_voice_query(query)
                
                # Give extra time for any background threads to complete
                time.sleep(2)
                
                print(f"✅ Client {client_id} processing completed with {len(responses)} responses")
                self.emit_to_client(client_socket, "status", f"✅ Completed - Session {session_id}")
                
            finally:
                # Restore original speak
                chatbot.speak_text = original_speak
                
        except Exception as e:
            logger.error(f"Processing error for client {client_id}: {e}")
            self.emit_to_client(client_socket, "error", str(e))
            
    def start_server(self):
        """Start the server."""
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            print(f"🚀 Individual Session Socket Server on {self.host}:{self.port}")
            print("👥 Each client gets their own session")
            print("🗑️ Sessions expire when clients disconnect")
            print("-" * 50)
            
            while True:
                client_socket, client_address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
        finally:
            self.server_socket.close()

def main():
    server = SimplifiedSocketServer()
    server.start_server()

if __name__ == "__main__":
    main()