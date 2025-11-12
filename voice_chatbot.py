#!/usr/bin/env python3
"""
Voice AI Chatbot with Smart Caching and Parallel Processing
Flow: Voice → Text → Cache/Greeting Check → Quick Response (TTS) + Parallel RAG → Detailed Response (TTS)
Modified: Automatic 8-second voice recordings in continuous loop
"""

import pyttsx3
import speech_recognition as sr
import logging
from typing import Optional
from dotenv import load_dotenv

from main import RAGOrchestrator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceChatbot:
    """Voice chatbot with smart caching and parallel RAG processing."""
    
    def __init__(self):
        """Initialize the Voice Chatbot."""
        print("🚀 Initializing Voice Chatbot...")
        
        # Initialize the RAG orchestrator directly
        self.rag_orchestrator = RAGOrchestrator()
        self.session_id = None
        
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Optimize recognizer settings for better speech detection
        self.recognizer.energy_threshold = 300  # Lower threshold for better sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Longer pause threshold
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5  # Wait for quiet before starting
        
        # Initialize TTS with better error handling
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            print("✅ TTS engine initialized successfully")
        except Exception as e:
            print(f"⚠️ TTS initialization warning: {e}")
            self.tts_engine = None  # Will be handled in speak_text method
        
        # Cache statistics (for tracking)
        self.cache_hits = 0
        
        # Quick ambient noise adjustment
        print("🎙️ Adjusting for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Test TTS functionality
        print("🎤 Testing TTS functionality...")
        self.speak_text("Voice chatbot initialized and ready!")
        
        print("✅ Voice Chatbot initialized successfully!")
    
    def record_and_transcribe(self) -> Optional[str]:
        """Record audio for exactly 8 seconds and convert to text using speech recognition."""
        try:
            # Ensure TTS is completely stopped before input
            import time
            time.sleep(1.0)  # Longer delay to ensure TTS has finished
            
            print("🎙️ Get ready! Starting 8-second recording in...")
            for i in range(1, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            
            print("🔴 RECORDING NOW! Speak clearly for 8 seconds...")
            
            # Record audio for exactly 8 seconds - no progress indicator to avoid interference
            with self.microphone as source:
                # Quick ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                
                # Record for exactly 8 seconds with no timeout issues
                start_time = time.time()
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8.0)
                end_time = time.time()
                
                actual_duration = end_time - start_time
                print(f"⏹️ Recording complete! ({actual_duration:.1f}s)")
            
            if audio is None:
                print("🤔 No audio recorded. Please try again.")
                return None
            
            print("🧠 Converting speech to text...")
            
            # Convert speech to text using Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio, show_all=False)
                if text:
                    print(f"📝 You said: '{text}'")
                    return text.strip()
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                return None
            except Exception:
                pass
            
            # Fallback: try with show_all=True for better results
            try:
                result = self.recognizer.recognize_google(audio, show_all=True)
                if result and 'alternative' in result and result['alternative']:
                    text = result['alternative'][0]['transcript']
                    print(f"📝 You said: '{text}'")
                    return text.strip()
            except Exception as e:
                logger.debug(f"Fallback recognition failed: {e}")
            
            print("🤔 Could not understand the audio. Please speak more clearly and closer to the microphone.")
            return None
                
        except sr.WaitTimeoutError:
            print("🤔 No speech detected. Please make sure your microphone is working.")
            return None
        except KeyboardInterrupt:
            print("\n🛑 Recording cancelled by user.")
            return "exit"
        except Exception as e:
            logger.error(f"Failed to record/transcribe: {e}")
            print("❌ Voice input failed. Please check your microphone and try again.")
            return None
    
    def speak_text(self, text: str) -> None:
        """Convert text to speech using the most reliable method available."""
        import subprocess
        import shlex
        
        if not text or not text.strip():
            print("⚠️ No text to speak")
            return
            
        print(f"🔊 SPEAKING: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Use macOS system 'say' command as primary method - most reliable
        try:
            # Escape the text properly for shell
            escaped_text = shlex.quote(text)
            
            # Use macOS built-in 'say' command with rate control (no timeout)
            result = subprocess.run(
                ['say', '-r', '150', escaped_text], 
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ SPEECH COMPLETED!")
            return
            
        except subprocess.CalledProcessError as e:
            print(f"❌ System say command failed: {e}")
        except FileNotFoundError:
            print("❌ 'say' command not found (not macOS?)")
        except Exception as e:
            print(f"❌ System TTS failed: {e}")
        
        # Fallback to pyttsx3 only if system TTS fails
        try:
            print("🎤 Falling back to pyttsx3...")
            
            # Create fresh engine each time to avoid state issues
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)  # Maximum volume
            
            engine.say(text)
            engine.runAndWait()
            
            # Clean up
            del engine
            
            print("✅ FALLBACK SPEECH COMPLETED!")
            
        except Exception as e:
            print(f"❌ ALL TTS METHODS FAILED: {e}")
            print(f"📝 Text that couldn't be spoken: {text}")
    

    
    def process_voice_query(self, user_query: str) -> bool:
        """Process voice query using the original SimpleChatbot logic."""
        try:
            # Check for exit commands first (quick detection)
            exit_keywords = ['exit', 'quit', 'goodbye', 'bye', 'stop', 'thanks', 'thank you', 'that\'s all', 'done', 'finished']
            if any(keyword in user_query.lower() for keyword in exit_keywords):
                self.speak_text("Thank you for using TechGropse AI Assistant. Have a great day! Goodbye!")
                return False
            
            # IMMEDIATE CACHE CHECK - Check Redis cache right after transcription
            print(f"\n🔍 Checking cache for immediate response...")
            
            # Check immediate cache first (session-aware only)
            cached_response = self._check_immediate_cache(user_query)
            if cached_response:
                # CACHE HIT - Provide immediate cached response with personalization
                print("💾 Cache hit! Providing immediate session cached response...")
                personalized_response = f"As I mentioned earlier in our conversation, {cached_response}"
                print(f"🤖 Cached Response: {personalized_response}")
                self.speak_text(personalized_response)
                self.cache_hits += 1
                return True
            
            # CACHE MISS - Continue with normal flow
            print("🔄 Cache miss. Proceeding with full RAG processing...")
            
            # CHECK FOR AMBIGUOUS/PARTIAL QUESTIONS - Stop processing if incomplete
            print("🔍 Checking if question is complete and clear...")
            ambiguity_check = self._check_question_completeness(user_query)
            
            if ambiguity_check.get("is_incomplete"):
                # INCOMPLETE QUESTION - Ask for clarification and stop
                print("⚠️ Question seems incomplete or ambiguous. Asking for clarification...")
                clarification_request = ambiguity_check.get("clarification_request")
                print(f"🤖 Clarification Request: {clarification_request}")
                self.speak_text(clarification_request)
                return True  # Don't proceed to RAG processing
            
            print("✅ Question is clear. Proceeding with RAG processing...")
            
            # Single LLM call to check if it's a greeting or ending
            engaging_response = self._check_and_generate_engaging_response(user_query)
            
            # Check if user wants to end the conversation
            if engaging_response == "CONVERSATION_ENDING":
                print("👋 User indicated conversation ending...")
                self.speak_text("Thank you for using TechGropse AI Assistant! It was great helping you today. Have a wonderful day! Goodbye!")
                return False
            
            if engaging_response:
                # NOT a greeting - use parallel processing
                import threading
                import queue
                
                # Start RAG pipeline in background thread
                response_queue = queue.Queue()
                
                def process_in_background():
                    try:
                        response = self.rag_orchestrator.process_query(
                            user_query=user_query,
                            session_id=self.session_id
                        )
                        response_queue.put(response)
                    except Exception as e:
                        logger.error(f"Background RAG processing failed: {e}")
                        response_queue.put(None)
                
                print("🚀 Starting RAG pipeline in background...")
                rag_thread = threading.Thread(target=process_in_background, daemon=True)
                rag_thread.start()
                
                # Speak engaging response while RAG processes in parallel
                print(f"🚀 Quick Response: {engaging_response}")
                print("🔊 Speaking quick response while processing your request...")
                self.speak_text(engaging_response)
                
                # Check if RAG completed while we were speaking
                try:
                    response = response_queue.get(timeout=5)  # Check if ready
                    if response and response.get("response"):
                        # RAG completed - speak the detailed response
                        detailed_response = response["response"]
                        print(f"🤖 Detailed Response: {detailed_response}")
                        print("🔊 Here's the detailed information...")
                        self.speak_text(detailed_response)
                        
                        # Update cache hits counter if it was a cache hit
                        if response.get("cache_hit"):
                            self.cache_hits += 1
                    else:
                        print("⚠️ Could not get detailed response")
                        
                except queue.Empty:
                    # RAG still processing - let user know
                    print("⏳ Still gathering detailed information...")
                    wait_msg = "I'm still gathering comprehensive details for you. Please give me a moment."
                    self.speak_text(wait_msg)
                    
                    # Wait a bit more for RAG completion
                    try:
                        response = response_queue.get(timeout=30)
                        if response and response.get("response"):
                            detailed_response = response["response"]
                            print(f"🤖 Detailed Response: {detailed_response}")
                            print("🔊 Thank you for waiting. Here are the details...")
                            self.speak_text(f"Thank you for waiting. {detailed_response}")
                            
                            if response.get("cache_hit"):
                                self.cache_hits += 1
                        else:
                            fallback_msg = "I apologize, but I'm having trouble accessing detailed information right now. Please try asking again."
                            self.speak_text(fallback_msg)
                    except queue.Empty:
                        timeout_msg = "I apologize for the delay. Please try asking your question again."
                        print("⚠️ RAG processing timeout")
                        self.speak_text(timeout_msg)
                
                return True
                
            else:
                # IS a greeting - process normally (no parallel processing needed)
                print("👋 Processing greeting...")
                response = self.rag_orchestrator.process_query(
                    user_query=user_query,
                    session_id=self.session_id
                )
                
                if response and response.get("response"):
                    response_text = response["response"]
                    print(f"🤖 Greeting Response: {response_text}")
                    self.speak_text(response_text)
                    
                    if response.get("cache_hit"):
                        self.cache_hits += 1
                        
                    return True
                else:
                    error_msg = "Hello! How can I help you with TechGropse information today?"
                    print(f"🤖 Fallback Greeting: {error_msg}")
                    self.speak_text(error_msg)
                    return True
            
        except Exception as e:
            logger.error(f"Voice query processing failed: {e}")
            error_msg = "I apologize, but I encountered an error processing your request. Please try again."
            self.speak_text(error_msg)
            return True
    
    def _check_immediate_cache(self, user_query: str) -> str:
        """
        Check session chat history for previous Q&A pairs using semantic matching with LLM.
        
        Args:
            user_query: User query string
            
        Returns:
            Cached response text from session if found, None otherwise
        """
        try:
            redis_manager = self.rag_orchestrator.redis_manager
            
            # Check session chat history for previous Q&A pairs
            session_history = redis_manager.get_chat_history(self.session_id)
            session_queries_and_responses = []
            
            if not session_history:
                print("🔍 No session history found")
                return None
                
            print(f"🔍 Checking {len(session_history)} messages in current session...")
            
            # Extract Q&A pairs from session history
            for i, message in enumerate(session_history):
                if message.get("role") == "user" and i + 1 < len(session_history):
                    user_msg = message.get("content", "")
                    next_msg = session_history[i + 1]
                    if next_msg.get("role") == "assistant":
                        assistant_response = next_msg.get("content", "")
                        if user_msg and assistant_response and len(assistant_response) > 30:
                            session_queries_and_responses.append({
                                "query": user_msg,
                                "response": assistant_response
                            })
            
            if not session_queries_and_responses:
                print("🔍 No valid Q&A pairs found in session history")
                return None
            
            # Use LLM for semantic matching within session
            print(f"🧠 Checking semantic similarity against {len(session_queries_and_responses)} session Q&A pairs...")
            
            semantic_match = self._find_semantic_match(user_query, session_queries_and_responses)
            
            if semantic_match:
                print(f"✅ Found semantic match in session history!")
                return semantic_match
            
            print("❌ No semantic match found in session history")
            return None
            
        except Exception as e:
            logger.error(f"Session cache check failed: {e}")
            return None

    def _find_semantic_match(self, user_query: str, cached_queries_and_responses: list) -> str:
        """
        Use LLM to find semantically similar cached query.
        
        Args:
            user_query: Current user query
            cached_queries_and_responses: List of cached queries and responses
            
        Returns:
            Cached response if semantic match found, None otherwise
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick semantic matching
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=100,
                timeout=3
            )
            
            # Prepare cached queries text for LLM analysis (limit to 8 for speed)
            limited_queries = cached_queries_and_responses[:8]
            cached_queries_text = ""
            for i, item in enumerate(limited_queries):
                # Truncate long queries for efficiency
                query_preview = item['query'][:100] + "..." if len(item['query']) > 100 else item['query']
                cached_queries_text += f"{i+1}. \"{query_preview}\"\n"
            
            system_prompt = f"""You are an expert at semantic similarity matching. Determine if the current user query is asking for the SAME information as any cached query.

CACHED QUERIES:
{cached_queries_text}

RULES:
- If the current query asks for the SAME core information as any cached query (even with different wording), respond with the NUMBER
- If NO semantic match exists, respond with "0"
- Be strict: only match if they want the same information

Examples:
- "What is your privacy policy?" ≈ "Tell me about privacy" ≈ "Privacy details" → MATCH
- "Hello" ≠ "What is privacy policy?" → NO MATCH"""

            user_prompt = f'Current: "{user_query}"\nResponse (number or 0):'

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            # Parse LLM response
            try:
                match_number = int(result)
                if 1 <= match_number <= len(limited_queries):
                    matched_item = limited_queries[match_number - 1]
                    print(f"🎯 Semantic match found: \"{matched_item['query'][:60]}...\"")
                    return matched_item['response']
                elif match_number == 0:
                    return None
            except ValueError:
                print(f"⚠️ Could not parse LLM response: {result}")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return None
    
    def _check_question_completeness(self, user_query: str) -> dict:
        """
        Check if the user's question is complete and clear enough to process.
        
        Args:
            user_query: User query string
            
        Returns:
            Dict with 'is_incomplete' boolean and 'clarification_request' string
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick analysis
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=100,
                timeout=5
            )
            
            system_prompt = """You are an AI assistant for TechGropse, a leading app development company.

Analyze if the user query is complete and clear enough to provide a meaningful response.

A query is INCOMPLETE/AMBIGUOUS if:
- Contains only partial words or fragments (like "what", "how", "child", "cookies")
- Is extremely vague without context (like "tell me about it", "what about this?")
- Has unclear pronouns without antecedents ("What about them?", "How does it work?")
- Is just a single word topic without a clear question

A query is COMPLETE if:
- Has a clear question structure ("What are the types of cookies?")
- Provides enough context to understand intent ("How do you handle privacy?")
- Is a proper greeting ("Hello", "Hi there")

RESPOND with either:
- "COMPLETE" if the query is clear enough
- "INCOMPLETE: [specific clarification request]" if it needs more information

Examples:
- "cookies" → "INCOMPLETE: Could you please clarify what you'd like to know about cookies? Are you asking about our cookie policy, types of cookies we use, or something else?"
- "what are cookies?" → "COMPLETE"
- "child" → "INCOMPLETE: I'd be happy to help with child-related information. Could you please specify what you'd like to know about children? Are you asking about child safety, privacy policies for minors, or something else?"
- "how does it work?" → "INCOMPLETE: Could you please specify what you're referring to? I'd be happy to explain how our services, policies, or features work if you could provide more context."
"""

            user_prompt = f'Query: "{user_query}"\nResponse:'

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            if result == "COMPLETE":
                return {"is_incomplete": False, "clarification_request": None}
            elif result.startswith("INCOMPLETE:"):
                clarification = result.replace("INCOMPLETE:", "").strip()
                return {"is_incomplete": True, "clarification_request": clarification}
            else:
                # Fallback - assume complete if unclear
                return {"is_incomplete": False, "clarification_request": None}
            
        except Exception as e:
            logger.error(f"Question completeness check failed: {e}")
            # On error, assume complete to avoid blocking
            return {"is_incomplete": False, "clarification_request": None}

    def _check_and_generate_engaging_response(self, user_query: str) -> str:
        """
        Single LLM call to check if query is greeting and generate engaging response if not.
        
        Returns:
            Engaging response string if not a greeting, empty string if greeting
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick response
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=60,  # ~30-40 words for 12-15 seconds
                timeout=5
            )
            
            system_prompt = f"""You are an AI assistant for TechGropse, a leading app development company.

Analyze the user query: "{user_query}"

TASK:
1. If it's a simple greeting (hi, hello, how are you, etc.) → respond with: "GREETING"
2. If it's ending the conversation (thanks, thank you, bye, goodbye, that's all, done, finished, etc.) → respond with: "ENDING"
3. If it's NOT a greeting or ending → generate a brief engaging response (45-60 words, 12-15 seconds of speech) that:
   - Acknowledges their question warmly
   - Shows TechGropse's expertise in that area  
   - Indicates you're gathering detailed information
   - Keeps them engaged while processing

Examples:
- "hi" → "GREETING"
- "how are you" → "GREETING"
- "thanks" → "ENDING"
- "thank you" → "ENDING"
- "bye" → "ENDING"
- "that's all" → "ENDING"
- "I'm done" → "ENDING"
- "what is your privacy policy" → "Great question about privacy! At TechGropse, we prioritize data protection in all our solutions. Let me gather our comprehensive privacy details for you..."
- "child safety measures" → "Excellent inquiry about child safety! TechGropse takes child protection seriously in our app development. I'm retrieving detailed information about our safety measures..."

Be professional, conversational, and TechGropse-focused."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            # If it's a greeting, return empty string
            if result.upper() == "GREETING":
                return ""
            
            # If it's ending the conversation, return special marker
            if result.upper() == "ENDING":
                return "CONVERSATION_ENDING"
            
            # Otherwise return the engaging response
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate engaging response: {e}")
            # If LLM fails, assume it's not a greeting and provide fallback
            return "Thank you for your question! TechGropse has expertise in this area. Let me gather comprehensive information to provide you with the most accurate response."

    def start_voice_chat(self) -> None:
        """Start the main voice chat loop with automatic 8-second recordings."""
        print("\n" + "="*70)
        print("🎙️ TechGropse Voice AI Assistant - Automatic Mode")
        print("="*70)
        print("🗣️ Ask your questions about TechGropse!")
        print("⏰ System will automatically record for 8 seconds each time")
        print("🔄 Continuous recording after each response")
        print("🧠 Smart caching with parallel processing for optimal experience")
        print("🎯 Quick responses + detailed information")
        print("🛑 Say 'exit', 'quit', 'goodbye', 'thanks', or 'that's all' to end")
        print("="*70)
        
        # Create new session using RAG orchestrator's method
        self.session_id = self.rag_orchestrator.create_session()
        
        print(f"📱 Session ID: {self.session_id}")
        
        conversation_count = 0
        
        try:
            while True:
                conversation_count += 1
                print(f"\n🔄 Conversation {conversation_count}")
                print("-" * 50)
                
                # Ensure clean state before each conversation
                import sys
                import time
                sys.stdout.flush()
                sys.stdin.flush()
                time.sleep(0.5)  # Brief pause to ensure clean audio state
                
                # Automatic 8-second recording
                user_query = self.record_and_transcribe()
                if not user_query:
                    print("⚠️ No speech detected. Make sure you're speaking clearly into the microphone.")
                    print("🔄 Trying again in 1 second...")
                    time.sleep(1)
                    continue
                
                # Handle exit commands (fallback)
                exit_commands = ['exit', 'quit', 'goodbye', 'bye', 'stop', 'thanks', 'thank you']
                if user_query.lower() in exit_commands or any(cmd in user_query.lower() for cmd in ['that\'s all', 'i\'m done', 'finished']):
                    self.speak_text("Thank you for using TechGropse AI Assistant. Have a great day! Goodbye!")
                    break
                
                # Process query and get response
                should_continue = self.process_voice_query(user_query)
                if not should_continue:
                    break
                
                print(f"\n⏳ Response complete. Next recording starts in 1 second...")
                print("─" * 50)
                
                # Brief pause before next recording cycle
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Voice chat interrupted by user.")
            self.speak_text("Voice chat ended. Goodbye!")
            
        # Final statistics
        print("\n" + "="*70)
        print("📊 Session Statistics")
        print("="*70)
        
        print(f"💬 Total Conversations: {conversation_count - 1}")
        print(f"💾 Cache Hits: {self.cache_hits}")
        
        print(f"\n👋 Voice chat session ended. Thank you!")

def main():
    """Main function to start the voice chatbot."""
    try:
        chatbot = VoiceChatbot()
        chatbot.start_voice_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start voice chatbot: {e}")
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    main()