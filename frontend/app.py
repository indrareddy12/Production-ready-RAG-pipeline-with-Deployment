import os
import streamlit as st
import requests
import uuid
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from http import HTTPStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors."""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DocumentChatApp:
    def __init__(self, backend_url: str = "http://backend:8000", timeout: int = 1000):
        """
        Initialize the Streamlit document chat application.
        
        Args:
            backend_url (str): Base URL for the backend API
            timeout (int): Request timeout in seconds
        """
        self.BACKEND_URL = backend_url.rstrip('/')
        self.TIMEOUT = timeout
        self.SUPPORTED_FORMATS = ["pdf", "docx", "txt"]
        self._initialize_session_state()

    def _make_api_request(
        self, 
        endpoint: str, 
        method: str = "POST", 
        **kwargs
    ) -> Tuple[Dict, int]:
        """
        Make an API request with error handling.
        
        Args:
            endpoint (str): API endpoint
            method (str): HTTP method
            **kwargs: Additional request parameters
        
        Returns:
            Tuple[Dict, int]: (Response data, status code)
        
        Raises:
            APIError: If the request fails
        """
        try:
            url = f"{self.BACKEND_URL}/{endpoint.lstrip('/')}"
            kwargs.setdefault('timeout', self.TIMEOUT)
            
            response = requests.request(method, url, **kwargs)
            
            if response.status_code == HTTPStatus.OK:
                return response.json(), response.status_code
            
            error_msg = f"API request failed: {response.text}"
            logger.error(error_msg)
            raise APIError(error_msg, response.status_code)
            
        except requests.Timeout:
            error_msg = f"Request to {endpoint} timed out after {self.TIMEOUT}s"
            logger.error(error_msg)
            raise APIError(error_msg)
            
        except requests.ConnectionError:
            error_msg = f"Cannot connect to backend server at {self.BACKEND_URL}"
            logger.error(error_msg)
            raise APIError(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg)
    
    def check_health(self) -> Dict:
        """
        Check backend health status.
        
        Returns:
            Dict: Health check response
        """
        try:
            data, status_code = self._make_api_request(
                'health',
                method="GET"
            )
            return {
                "status": "healthy" if status_code == HTTPStatus.OK else "unhealthy",
                "backend_status": data.get("status", "unknown"),
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat())
            }
        except APIError as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def format_metrics(self, raw_metrics: str) -> str:
        """Format metrics for better display"""
        formatted = []
        for line in raw_metrics.split('\n'):
            if line.startswith('# HELP'):
                metric_name = line.split()[2]
                description = ' '.join(line.split()[3:])
                formatted.append(f"\n## {metric_name}")
                formatted.append(f"Description: {description}")
            elif line.startswith('# TYPE'):
                metric_type = line.split()[-1]
                formatted.append(f"Type: {metric_type}\n")
            elif line and not line.startswith('#'):
                name, value = line.rsplit(' ', 1)
                formatted.append(f"- {name}: {value}")
        
        return '\n'.join(formatted)

    def get_metrics(self):
        try:
            response = requests.get(
                f"{self.BACKEND_URL}/metrics",
                timeout=self.TIMEOUT
            )
            
            if response.status_code == HTTPStatus.OK:
                return self.format_metrics(response.text)
                
            st.error(f"Failed to fetch metrics: Status {response.status_code}")
            return None
            
        except requests.RequestException as e:
            st.error(f"Error fetching metrics: {str(e)}")
            return None

    def upload_files(self, uploaded_files: List) -> bool:
        """
        Upload files to the backend.
        
        Args:
            uploaded_files (List): List of uploaded files
        
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            if not uploaded_files:
                return False

            file_data = [("files", (file.name, file.getvalue())) for file in uploaded_files]
            
            _, status_code = self._make_api_request(
                'upload/',
                files=file_data,
                data={"session_id": st.session_state["session_id"]}
            )
            
            if status_code == HTTPStatus.OK:
                st.success("Files uploaded successfully!")
                st.session_state["uploaded_files"] = uploaded_files
                return True
                
            return False
            
        except APIError as e:
            st.error(e.message)
            return False

    def get_chat_history(self) -> List[Dict]:
        """
        Fetch chat history from backend.
        
        Returns:
            List[Dict]: List of chat history entries
        """
        try:
            data, _ = self._make_api_request(
                'chat_history/',
                json={"session_id": st.session_state["session_id"]}
            )
            return data.get("chat_history", [])
            
        except APIError as e:
            logger.error(f"Error fetching chat history: {e}")
            return []

    def submit_query(self, query: str) -> Optional[str]:
        """
        Submit a query to the backend and update chat history immediately.
        
        Args:
            query (str): User query
            
        Returns:
            Optional[str]: Response text if successful, None otherwise
        """
        try:
            data, _ = self._make_api_request(
                'query/',
                json={
                    "session_id": st.session_state["session_id"],
                    "query": query
                },
                headers={"Content-Type": "application/json"}
            )
            
            if "error" in data:
                st.error(data["error"])
                return None
            
            response_text = data.get("response", "No response received.")
            self._update_chat_history(query, response_text)
            return response_text
            
        except APIError as e:
            st.error(e.message)
            return None

    def _update_chat_history(self, question: str, answer: str):
        """
        Update session state with new chat message.
        
        Args:
            question (str): User question
            answer (str): Bot answer
        """
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        st.session_state.chat_messages.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })

    def render_chat_history(self):
        """Render the chat history in the Streamlit app."""
        st.subheader("Chat History")
        
        if not st.session_state.get('chat_messages') and not self.get_chat_history():
            st.markdown("*No chat history yet.*")
            return
            
        chat_container = st.container()
        
        with chat_container:
            # First show older messages from backend
            for chat in self.get_chat_history():
                try:
                    chat_data = json.loads(chat)
                    if not any(m['question'] == chat_data['question'] 
                            for m in st.session_state.get('chat_messages', [])):
                        with st.container():
                            st.markdown(f"**You:** {chat_data['question']}")
                            st.markdown(f"**Bot:** {chat_data['answer']}")
                            st.markdown("---")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing chat history: {e}")
            
            # Then show current session messages in chronological order
            if st.session_state.get('chat_messages'):
                for chat in st.session_state.chat_messages:  # Removed reversed()
                    with st.container():
                        st.markdown(f"**You:** {chat['question']}")
                        st.markdown(f"**Bot:** {chat['answer']}")
                        st.markdown("---")

    def end_session(self):
        """End the current session and reset session state."""
        try:
            self._make_api_request(
                'cleanup/',
                json={"session_id": st.session_state["session_id"]}
            )
        except APIError as e:
            logger.warning(f"Error during session cleanup: {e}")
        finally:
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            self._initialize_session_state()
            st.rerun()

    def _initialize_session_state(self):
        """Initialize or reset session state variables."""
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())
        
        session_state_defaults = {
            "answer": "",
            "query": "",
            "last_query": "",
            "uploaded_files": [],
            "error": None,
            "chat_messages": [],
            "input_key": 0  # Add this line
        }
        
        for key, default_value in session_state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def run(self):
        """Main method to run the Streamlit application."""
        st.set_page_config(
            page_title="Chat with Documents",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add system status to sidebar
        with st.sidebar:
            st.subheader("System Status")
            
            # Health check
            health_status = self.check_health()
            if health_status["status"] == "healthy":
                st.success("Backend: Healthy")
            else:
                st.error("Backend: Unhealthy")
                if "error" in health_status:
                    st.warning(f"Error: {health_status['error']}")
            
            # Last updated timestamp
            st.caption(f"Last checked: {datetime.fromisoformat(health_status['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Add metrics display option
            if st.checkbox("Show System Metrics"):
                metrics_data = self.get_metrics()
                if metrics_data:
                    with st.expander("System Metrics", expanded=True):
                        st.text(metrics_data)
                else:
                    st.warning("Unable to fetch metrics")
            
            # Add divider
            st.divider()
        
        st.title("Chat with Documents")
        
        # File upload section
        with st.expander("📁 Upload Documents (Optional)"):
            st.markdown(f"""
            You can upload documents to get contextual responses, or just chat without documents.
            Supported formats: {', '.join(self.SUPPORTED_FORMATS)}
            """)
            
            uploaded_files = st.file_uploader(
                "Upload files",
                type=self.SUPPORTED_FORMATS,
                accept_multiple_files=True
            )

            if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
                self.upload_files(uploaded_files)

        # Chat interface
        st.markdown("### Chat")
        if st.session_state.get("uploaded_files"):
            st.info(f"Chatting with {len(st.session_state['uploaded_files'])} uploaded documents")
        else:
            st.info("Chat without documents - responses will be based on general knowledge")

        # Add clear chat button and session info
        col1, col2 = st.columns([6, 4])
        with col1:
            if st.session_state.get("chat_messages"):
                if st.button("Clear Chat History"):
                    st.session_state.chat_messages = []
                    st.session_state.last_query = ""
                    st.session_state.input_key += 1
                    st.rerun()
        with col2:
            st.caption(f"Session ID: {st.session_state['session_id']}")

        # Chat container for history
        chat_container = st.container()
        with chat_container:
            self.render_chat_history()

        # Input container for user interaction
        input_container = st.container()
        with input_container:
            user_input = st.text_input(
                "Ask a question",
                key=f"text_input_{st.session_state.input_key}"
            )
            
            if user_input and user_input != st.session_state.get("last_query"):
                with st.spinner('Processing your question...'):
                    response = self.submit_query(user_input)
                    
                    if response:
                        st.session_state["last_query"] = user_input
                        st.session_state.input_key += 1
                        st.rerun()

        # Session control
        if st.button("End Session"):
            self.end_session()

def main():
    """Main entry point for the application."""
    try:
        backend_url = os.getenv("BACKEND_URL")
        app = DocumentChatApp(backend_url=backend_url)
        app.run()
    except Exception as e:
        logger.exception("Application failed to start")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()