#!/usr/bin/env python3
"""
Simple Socket Client for testing Session Voice Server
Shows real-time responses as they are emitted
"""

import socket
import json
import threading
import time

class SocketVoiceClient:
    """Simple client for testing the session socket voice server."""
    
    def __init__(self, host='localhost', port=8889):
        """Initialize the client."""
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            print(f"🔗 Connected to {self.host}:{self.port}")
            print("=" * 50)
            
            # Start listening for responses
            listener_thread = threading.Thread(target=self.listen_for_responses, daemon=True)
            listener_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
            
    def listen_for_responses(self):
        """Listen for responses from server."""
        buffer = ""
        
        while self.connected:
            try:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                
                # Process complete messages (one per line)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            self.handle_server_message(message)
                        except json.JSONDecodeError:
                            print(f"📨 Raw: {line}")
                            
            except Exception as e:
                if self.connected:
                    print(f"❌ Listen error: {e}")
                break
                
    def handle_server_message(self, message):
        """Handle incoming server messages."""
        msg_type = message.get('type', 'unknown')
        data = message.get('data', '')
        timestamp = message.get('timestamp', '')
        
        # Format different message types
        if msg_type == 'connected':
            print(f"[{timestamp}] 🎉 {data}")
        elif msg_type == 'status':
            print(f"[{timestamp}] ℹ️  {data}")
        elif msg_type == 'query_received':
            print(f"[{timestamp}] 🗣️  {data}")
        elif msg_type == 'response':
            print(f"[{timestamp}] 🤖 {data}")
        elif msg_type == 'error':
            print(f"[{timestamp}] ❌ ERROR: {data}")
        else:
            print(f"[{timestamp}] 📨 {msg_type.upper()}: {data}")
            
    def send_query(self, query):
        """Send query to server."""
        if not self.connected:
            print("❌ Not connected to server")
            return
            
        try:
            message = {"query": query}
            json_message = json.dumps(message) + '\n'
            self.socket.send(json_message.encode('utf-8'))
            print(f"📤 Sent: {query}")
            
        except Exception as e:
            print(f"❌ Send error: {e}")
            
    def disconnect(self):
        """Disconnect from server."""
        self.connected = False
        if self.socket:
            self.socket.close()
        print("🔌 Disconnected")
        
    def interactive_session(self):
        """Run interactive session."""
        print("🎯 Interactive Session with Session-Based Cache")
        print("💡 Type your questions or 'quit' to exit")
        print("🔄 Ask the same question twice to test session cache!")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n🗣️  Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    self.send_query("goodbye")
                    time.sleep(1)
                    break
                    
                if query:
                    self.send_query(query)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session ended")
                break
                
        self.disconnect()

def main():
    """Main function."""
    print("🚀 Session-Based Socket Voice Client")
    print("=" * 35)
    
    client = SocketVoiceClient()
    
    if client.connect():
        time.sleep(1)  # Wait for connection message
        client.interactive_session()
    else:
        print("❌ Failed to connect to server")

if __name__ == "__main__":
    main()