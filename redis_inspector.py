#!/usr/bin/env python3
"""
Redis Cache Inspector for the AI Agentic RAG system.
This script allows you to inspect Redis cache contents, session data, and cache performance.
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

from config import REDIS_CONFIG, REDIS_KEYS
from utils.redis_manager import RedisManager


class RedisInspector:
    """Inspector for Redis cache system."""
    
    def __init__(self):
        """Initialize the Redis inspector."""
        self.redis_manager = RedisManager()
        print(f"✅ Connected to Redis at {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
    
    def show_redis_stats(self) -> None:
        """Display Redis server statistics."""
        print("\n" + "="*60)
        print("🔍 REDIS SERVER STATISTICS")
        print("="*60)
        
        try:
            info = self.redis_manager.client.info()
            
            print(f"Redis Version: {info.get('redis_version', 'N/A')}")
            print(f"Connected Clients: {info.get('connected_clients', 0)}")
            print(f"Used Memory: {info.get('used_memory_human', 'N/A')}")
            print(f"Total Commands Processed: {info.get('total_commands_processed', 0):,}")
            print(f"Keyspace Hits: {info.get('keyspace_hits', 0):,}")
            print(f"Keyspace Misses: {info.get('keyspace_misses', 0):,}")
            
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            print(f"Cache Hit Rate: {hit_rate:.1f}%")
            
            print(f"Uptime: {info.get('uptime_in_seconds', 0):,} seconds")
            
            # Database info
            db_info = info.get(f'db{REDIS_CONFIG["db"]}', {})
            if db_info:
                print(f"\nDatabase {REDIS_CONFIG['db']} Info:")
                print(f"  Keys: {db_info.get('keys', 0)}")
                print(f"  Expires: {db_info.get('expires', 0)}")
                
        except Exception as e:
            print(f"❌ Error getting Redis stats: {e}")
    
    def list_all_keys(self, pattern: str = "*") -> List[str]:
        """List all keys matching a pattern."""
        try:
            keys = self.redis_manager.client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            print(f"❌ Error listing keys: {e}")
            return []
    
    def show_cache_overview(self) -> None:
        """Show overview of cached data."""
        print("\n" + "="*60)
        print("📊 CACHE OVERVIEW")
        print("="*60)
        
        # Get all keys
        all_keys = self.list_all_keys()
        
        if not all_keys:
            print("🔍 No keys found in Redis cache")
            return
        
        # Categorize keys
        sessions = [k for k in all_keys if k.startswith("chat:")]
        intents = [k for k in all_keys if k.startswith("intent:")]
        session_metas = [k for k in all_keys if k.startswith("session_meta:")]
        doc_caches = [k for k in all_keys if k.startswith("docs_cache:")]
        summary_caches = [k for k in all_keys if k.startswith("summary_cache:")]
        locks = [k for k in all_keys if k.startswith("locks:")]
        
        print(f"📝 Total Keys: {len(all_keys)}")
        print(f"💬 Chat Sessions: {len(sessions)}")
        print(f"🎯 Intent Classifications: {len(intents)}")
        print(f"📋 Session Metadata: {len(session_metas)}")
        print(f"📄 Document Caches: {len(doc_caches)}")
        print(f"📝 Summary Caches: {len(summary_caches)}")
        print(f"🔒 Active Locks: {len(locks)}")
        
        if sessions:
            print(f"\n🔍 Recent Sessions:")
            for session in sessions[-5:]:  # Show last 5 sessions
                session_id = session.replace("chat:", "")
                print(f"  • {session_id}")
    
    def inspect_session(self, session_id: str) -> None:
        """Inspect a specific session."""
        print("\n" + "="*60)
        print(f"🔍 SESSION INSPECTION: {session_id}")
        print("="*60)
        
        # Chat history
        print("\n💬 CHAT HISTORY:")
        chat_history = self.redis_manager.get_chat_history(session_id)
        if chat_history:
            for i, message in enumerate(chat_history, 1):
                timestamp = message.get('timestamp', 'Unknown')
                role = message.get('role', 'unknown')
                content = message.get('content', '')[:100]
                print(f"  {i}. [{timestamp}] {role.upper()}: {content}{'...' if len(message.get('content', '')) > 100 else ''}")
        else:
            print("  No chat history found")
        
        # Session metadata
        print("\n📋 SESSION METADATA:")
        metadata = self.redis_manager.get_session_metadata(session_id)
        if metadata:
            print(f"  Created: {metadata.get('created_at', 'Unknown')}")
            print(f"  Last Active: {metadata.get('last_activity', 'Unknown')}")
            print(f"  Total Messages: {metadata.get('total_messages', 0)}")
            print(f"  Queries Processed: {metadata.get('queries_processed', 0)}")
            print(f"  Total Processing Time: {metadata.get('total_processing_time', 0):.2f}s")
            print(f"  User ID: {metadata.get('user_id', 'Anonymous')}")
        else:
            print("  No metadata found")
        
        # Intent history
        print("\n🎯 INTENT CLASSIFICATIONS:")
        intent_key = f"intent:{session_id}"
        try:
            intent_data = self.redis_manager.client.get(intent_key)
            if intent_data:
                intent_info = json.loads(intent_data)
                print(f"  Last Intent: {intent_info.get('intent', 'Unknown')}")
                print(f"  Confidence: {intent_info.get('confidence', 0):.2f}")
                print(f"  Reasoning: {intent_info.get('reasoning', 'N/A')}")
            else:
                print("  No intent data found")
        except Exception as e:
            print(f"  Error reading intent data: {e}")
    
    def show_document_cache(self) -> None:
        """Show document cache contents."""
        print("\n" + "="*60)
        print("📄 DOCUMENT CACHE")
        print("="*60)
        
        doc_keys = self.list_all_keys("docs_cache:*")
        
        if not doc_keys:
            print("🔍 No document caches found")
            return
        
        print(f"📊 Total Document Caches: {len(doc_keys)}")
        
        for i, key in enumerate(doc_keys[:5], 1):  # Show first 5
            print(f"\n📄 Cache {i}: {key}")
            try:
                cache_data = self.redis_manager.client.get(key)
                if cache_data:
                    cache_obj = json.loads(cache_data)
                    
                    # Handle different cache structures
                    if isinstance(cache_obj, dict):
                        docs = cache_obj.get('docs', cache_obj)
                        query = cache_obj.get('query', 'Unknown query')
                        print(f"  Query: {query[:50]}...")
                    else:
                        docs = cache_obj
                    
                    if isinstance(docs, list):
                        print(f"  Documents: {len(docs)}")
                        if docs and len(docs) > 0:
                            first_doc = docs[0]
                            if isinstance(first_doc, dict):
                                content = first_doc.get('content', '')
                                print(f"  First Doc Preview: {content[:100]}...")
                                print(f"  Source: {first_doc.get('source', 'Unknown')}")
                            else:
                                print(f"  First Doc: {str(first_doc)[:100]}...")
                    else:
                        print(f"  Data type: {type(docs)}")
                else:
                    print("  No data found")
            except Exception as e:
                print(f"  Error reading cache: {e}")
    
    def show_summary_cache(self) -> None:
        """Show summary cache contents."""
        print("\n" + "="*60)
        print("📝 SUMMARY CACHE")
        print("="*60)
        
        summary_keys = self.list_all_keys("summary_cache:*")
        
        if not summary_keys:
            print("🔍 No summary caches found")
            return
        
        print(f"📊 Total Summary Caches: {len(summary_keys)}")
        
        for key in summary_keys:
            intent = key.replace("summary_cache:", "")
            print(f"\n📝 Intent: {intent}")
            try:
                summary_data = self.redis_manager.client.get(key)
                if summary_data:
                    summary = json.loads(summary_data)
                    summary_text = summary if isinstance(summary, str) else str(summary)
                    print(f"  Summary: {summary_text[:200]}{'...' if len(summary_text) > 200 else ''}")
                else:
                    print("  No data found")
            except Exception as e:
                print(f"  Error reading summary: {e}")
    
    def clear_cache(self, pattern: str = None) -> None:
        """Clear cache entries."""
        if pattern:
            keys = self.list_all_keys(pattern)
            if keys:
                count = self.redis_manager.client.delete(*keys)
                print(f"✅ Cleared {count} keys matching pattern: {pattern}")
            else:
                print(f"🔍 No keys found matching pattern: {pattern}")
        else:
            # Clear all cache data but keep sessions
            patterns = ["docs_cache:*", "summary_cache:*", "locks:*"]
            total_cleared = 0
            for pattern in patterns:
                keys = self.list_all_keys(pattern)
                if keys:
                    count = self.redis_manager.client.delete(*keys)
                    total_cleared += count
            print(f"✅ Cleared {total_cleared} cache entries")
    
    def monitor_realtime(self, duration: int = 30) -> None:
        """Monitor Redis activity in real-time."""
        print(f"\n🔍 MONITORING REDIS ACTIVITY FOR {duration} SECONDS")
        print("="*60)
        print("Press Ctrl+C to stop monitoring early...")
        
        try:
            import time
            start_time = time.time()
            last_stats = self.redis_manager.client.info()
            
            while time.time() - start_time < duration:
                time.sleep(2)
                current_stats = self.redis_manager.client.info()
                
                # Calculate deltas
                commands_delta = current_stats.get('total_commands_processed', 0) - last_stats.get('total_commands_processed', 0)
                hits_delta = current_stats.get('keyspace_hits', 0) - last_stats.get('keyspace_hits', 0)
                misses_delta = current_stats.get('keyspace_misses', 0) - last_stats.get('keyspace_misses', 0)
                
                if commands_delta > 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Commands: +{commands_delta}, Hits: +{hits_delta}, Misses: +{misses_delta}")
                
                last_stats = current_stats
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
    
    def close(self) -> None:
        """Close Redis connection."""
        self.redis_manager.close()


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Redis Cache Inspector for AI RAG System")
    parser.add_argument("--session", "-s", help="Inspect specific session ID")
    parser.add_argument("--stats", action="store_true", help="Show Redis statistics")
    parser.add_argument("--overview", action="store_true", help="Show cache overview")
    parser.add_argument("--docs", action="store_true", help="Show document cache")
    parser.add_argument("--summaries", action="store_true", help="Show summary cache")
    parser.add_argument("--clear", help="Clear cache (pattern or 'all')")
    parser.add_argument("--monitor", type=int, help="Monitor Redis activity for N seconds")
    parser.add_argument("--all", action="store_true", help="Show all information")
    
    args = parser.parse_args()
    
    inspector = RedisInspector()
    
    try:
        if args.all or (not any([args.session, args.stats, args.overview, args.docs, args.summaries, args.clear, args.monitor])):
            # Show everything if no specific option is given
            inspector.show_redis_stats()
            inspector.show_cache_overview()
            inspector.show_document_cache()
            inspector.show_summary_cache()
        
        if args.stats:
            inspector.show_redis_stats()
        
        if args.overview:
            inspector.show_cache_overview()
        
        if args.session:
            inspector.inspect_session(args.session)
        
        if args.docs:
            inspector.show_document_cache()
        
        if args.summaries:
            inspector.show_summary_cache()
        
        if args.clear:
            if args.clear.lower() == "all":
                inspector.clear_cache()
            else:
                inspector.clear_cache(args.clear)
        
        if args.monitor:
            inspector.monitor_realtime(args.monitor)
    
    finally:
        inspector.close()


if __name__ == "__main__":
    main()