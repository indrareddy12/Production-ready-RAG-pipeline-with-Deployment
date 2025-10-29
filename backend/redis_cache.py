import os
import redis
import json
import threading
from vector_store import delete_session_embeddings
from logger_config import logger

# Get Redis connection details from environment variables with defaults
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')  # Uses 'redis' as default for Docker
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

try:
    # Initialize Redis with connection retry logic
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_timeout=5,  # Socket timeout
        retry_on_timeout=True,  # Retry on timeout
        max_connections=10  # Connection pool size
    )
    # Test the connection
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise

def save_session(session_id, files):
    """Save session details (file names) in Redis with a 30-minute TTL."""
    session_data = {"files": files}
    redis_client.setex(session_id, 1800, json.dumps(session_data))

def get_session(session_id):
    """Retrieve session details from Redis. If session exists, extend TTL."""
    data = redis_client.get(session_id)
    if data:
        redis_client.expire(session_id, 1800)  # Refresh TTL on user activity
        return json.loads(data)
    return None  # If TTL expired, session will be None

def delete_session(session_id):
    """Remove session from Redis and trigger embedding cleanup in Pinecone."""
    redis_client.delete(session_id)
    redis_client.delete(f"{session_id}:chat")  # Delete chat history from Redis
    delete_session_embeddings(session_id)  # Cleanup embeddings

def get_chat_history(session_id):
    """Retrieve chat history from Redis."""
    chat_key = f"{session_id}:chat"
    return redis_client.lrange(chat_key, 0, -1)  # Fetch all chat messages from Redis list

def save_chat_history(session_id, question, answer):
    """Store chat history in Redis and set expiration."""
    chat_key = f"{session_id}:chat"
    chat_entry = json.dumps({"question": question, "answer": answer})
    redis_client.rpush(chat_key, chat_entry)  # Append new chat entry to Redis list
    redis_client.expire(chat_key, 1800)  

def redis_key_expiry_listener():
    """Continuously listen for session expiration events and clean up embeddings."""
    pubsub = redis_client.pubsub()
    pubsub.psubscribe("__keyevent@0__:expired")  # Listen for key expiration events

    for message in pubsub.listen():
        if message["type"] == "pmessage":
            expired_key = message["data"]
            print(f"Session expired: {expired_key}. Cleaning up embeddings...")
            delete_session_embeddings(expired_key)

# Run expiry listener in a background thread
threading.Thread(target=redis_key_expiry_listener, daemon=True).start()