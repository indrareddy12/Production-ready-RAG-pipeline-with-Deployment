import os
from pathlib import Path
import yaml
from typing import Dict, Optional
from logger_config import logger

class PromptManager:
    """Manages prompt templates from files and defaults"""
    
    def __init__(self, prompt_dir: Optional[str] = None):
        """
        Initialize PromptManager
        
        Args:
            prompt_dir: Optional directory containing prompt YAML files
        """
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.prompts: Dict[str, str] = {}
        self._load_prompts()
        
        # Default system prompt as fallback
        self.default_system_prompt = """You are a helpful AI assistant. Follow these guidelines:
        1. Use the provided context from documents when available to answer questions
        2. Consider the chat history for context and continuity
        3. If no relevant context is found, answer based on general knowledge
        4. Always be clear, concise, and accurate
        5. If you're unsure, acknowledge the uncertainty"""

    def _load_prompts(self) -> None:
        """Load prompts from YAML files in prompt directory"""
        if not self.prompt_dir or not self.prompt_dir.exists():
            logger.info("No prompt directory found, using default prompts")
            return
            
        try:
            for prompt_file in self.prompt_dir.glob("*.yaml"):
                with open(prompt_file, 'r') as f:
                    prompts = yaml.safe_load(f)
                    if isinstance(prompts, dict):
                        self.prompts.update(prompts)
            logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompt_dir}")
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            
    def get_system_prompt(self) -> str:
        """Get system prompt from file or default"""
        return self.prompts.get('system_prompt', self.default_system_prompt)
        
    def get_prompt_template(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get specific prompt template by name"""
        return self.prompts.get(name, default)

    def format_prompt(self, 
                     query: str,
                     context: str,
                     history_text: str) -> str:
        """Format the complete prompt with all components"""
        system_prompt = self.get_system_prompt()
        
        prompt_template = self.get_prompt_template(
            'chat_template',
            # Default template if none found in files
            """System: {system_prompt}

                Past Conversation:
                {history_text}

                Relevant Context:
                {context}

                Current Question: {query}

                Please provide a helpful response based on the above context and conversation history."""
        )
        
        return prompt_template.format(
            system_prompt=system_prompt,
            history_text=history_text if history_text else "No prior conversation.",
            context=context,
            query=query
        )