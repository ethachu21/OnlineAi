"""
OpenAI Handler
Handles API calls to OpenAI GPT models
"""

import json
import logging
import requests
from typing import Dict, Any, List
from .base_handler import GenerationHandler
from config import config

logger = logging.getLogger(__name__)


class OpenAiHandler(GenerationHandler):
    """Handler for OpenAI GPT models with multiple model support"""
    
    def __init__(self):
        self.name = "openai"
        self.provider_config = config.get_provider('openai')
        self.models = self.provider_config.models if self.provider_config else []
        self.current_model_index = 0
        self.api_keys = self.provider_config.get_valid_api_keys() if self.provider_config else []
        self.current_key_index = 0
        self.base_url = self.provider_config.base_url if self.provider_config else "https://api.openai.com/v1"
    
    def _rotate_api_key(self):
        """Rotate to the next API key"""
        if len(self.api_keys) <= 1:
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return True
    
    def _rotate_model(self):
        """Rotate to the next model"""
        if len(self.models) <= 1:
            return False
        
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        # Reset to first API key when switching models
        self.current_key_index = 0
        return True
    
    def is_available(self) -> bool:
        """Check if OpenAI handler is properly configured"""
        return len(self.api_keys) > 0 and self.provider_config is not None
    
    def get_name(self) -> str:
        """Get the name of this handler"""
        return self.name
    
    def supports_streaming(self) -> bool:
        """Check if this handler supports streaming"""
        return True
    
    def set_preferred_model(self, model_name: str):
        """Set a specific model as preferred for this handler"""
        if model_name in self.models:
            # Move the preferred model to the front
            self.models.remove(model_name)
            self.models.insert(0, model_name)
            self.current_model_index = 0
    
    def _supports_system_role(self) -> bool:
        """OpenAI supports system roles natively"""
        return True
    
    def generate_stream(self, prompt: str, messages: List[Dict] = None):
        """Generate a streaming response using OpenAI API"""
        if not self.is_available():
            yield {
                "success": False,
                "error": "OpenAI handler not properly configured - no API keys available",
                "provider": self.name
            }
            return
        
        total_attempts = 0
        models_tried = []
        
        # Try each model
        for model_attempt in range(len(self.models)):
            current_model = self.models[self.current_model_index]
            models_tried.append(current_model)
            
            # Try each API key for this model
            key_attempts = 0
            max_key_attempts = len(self.api_keys)
            
            while key_attempts < max_key_attempts:
                total_attempts += 1
                try:
                    # Process messages to handle system roles
                    processed_messages = self._process_messages(messages or [])
                    
                    # Build the messages array for OpenAI
                    openai_messages = []
                    
                    if processed_messages:
                        for msg in processed_messages:
                            openai_messages.append({
                                "role": msg.get('role', 'user'),
                                "content": msg.get('content', '')
                            })
                    
                    # Add the current prompt
                    openai_messages.append({
                        "role": "user",
                        "content": prompt
                    })
                    
                    # Prepare the request
                    url = f"{self.base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": current_model,
                        "messages": openai_messages,
                        "max_tokens": config.MAX_TOKENS,
                        "temperature": config.TEMPERATURE,
                        "stream": True
                    }
                    
                    # Special handling for O1 models (they don't support all parameters)
                    if "o1" in current_model:
                        data = {
                            "model": current_model,
                            "messages": openai_messages,
                            "max_completion_tokens": config.MAX_TOKENS,
                            "stream": True
                        }
                    
                    # Make the streaming API call
                    response = requests.post(url, headers=headers, json=data, 
                                           timeout=config.API_TIMEOUT, stream=True)
                    
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                line_text = line.decode('utf-8')
                                if line_text.startswith('data: '):
                                    data_content = line_text[6:]
                                    if data_content.strip() == '[DONE]':
                                        break
                                    try:
                                        json_data = json.loads(data_content)
                                        if 'choices' in json_data and len(json_data['choices']) > 0:
                                            choice = json_data['choices'][0]
                                            if 'delta' in choice and 'content' in choice['delta']:
                                                content = choice['delta']['content']
                                                if content:
                                                    yield {
                                                        "success": True,
                                                        "response": content,
                                                        "provider": self.name,
                                                        "model": current_model,
                                                        "streaming": True
                                                    }
                                    except json.JSONDecodeError:
                                        continue
                        return
                    else:
                        logger.warning(f"OpenAI {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"OpenAI {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            # All keys failed for this model, try next model
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All OpenAI keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        # All models and keys failed
        yield {
            "success": False,
            "error": f"All OpenAI models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        }
    
    def generate(self, prompt: str, messages: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response using OpenAI API with model and key rotation"""
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenAI handler not properly configured - no API keys available",
                "provider": self.name
            }
        
        total_attempts = 0
        models_tried = []
        
        # Try each model
        for model_attempt in range(len(self.models)):
            current_model = self.models[self.current_model_index]
            models_tried.append(current_model)
            
            # Try each API key for this model
            key_attempts = 0
            max_key_attempts = len(self.api_keys)
            
            while key_attempts < max_key_attempts:
                total_attempts += 1
                try:
                    # Process messages to handle system roles
                    processed_messages = self._process_messages(messages or [])
                    
                    # Build the messages array for OpenAI
                    openai_messages = []
                    
                    if processed_messages:
                        for msg in processed_messages:
                            openai_messages.append({
                                "role": msg.get('role', 'user'),
                                "content": msg.get('content', '')
                            })
                    
                    # Add the current prompt
                    openai_messages.append({
                        "role": "user",
                        "content": prompt
                    })
                    
                    # Prepare the request
                    url = f"{self.base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": current_model,
                        "messages": openai_messages,
                        "max_tokens": config.MAX_TOKENS,
                        "temperature": config.TEMPERATURE
                    }
                    
                    # Special handling for O1 models (they don't support all parameters)
                    if "o1" in current_model:
                        data = {
                            "model": current_model,
                            "messages": openai_messages,
                            "max_completion_tokens": config.MAX_TOKENS
                        }
                    
                    # Make the API call
                    response = requests.post(url, headers=headers, json=data, 
                                           timeout=config.API_TIMEOUT)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            choice = json_data['choices'][0]
                            if 'message' in choice and 'content' in choice['message']:
                                content = choice['message']['content']
                                return {
                                    "success": True,
                                    "response": content,
                                    "provider": self.name,
                                    "model": current_model,
                                    "api_key_index": self.current_key_index,
                                    "total_api_keys": len(self.api_keys),
                                    "models_tried": models_tried,
                                    "total_attempts": total_attempts
                                }
                        
                        return {
                            "success": False,
                            "error": "No valid response from OpenAI",
                            "provider": self.name,
                            "model": current_model,
                            "response_data": json_data
                        }
                    else:
                        logger.warning(f"OpenAI {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"OpenAI {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            # All keys failed for this model, try next model
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All OpenAI keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        # All models and keys failed
        return {
            "success": False,
            "error": f"All OpenAI models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        } 