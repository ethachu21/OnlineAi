"""
Cohere Handler
Handles API calls to Cohere models with free tier support
"""

import json
import logging
import requests
from typing import Dict, Any, List
from .base_handler import GenerationHandler
from config import config

logger = logging.getLogger(__name__)


class CohereHandler(GenerationHandler):
    """Handler for Cohere models with free tier support"""
    
    def __init__(self):
        self.name = "cohere"
        self.provider_config = config.get_provider('cohere')
        self.models = self.provider_config.models if self.provider_config else []
        self.current_model_index = 0
        self.api_keys = self.provider_config.get_valid_api_keys() if self.provider_config else []
        self.current_key_index = 0
        self.base_url = self.provider_config.base_url if self.provider_config else "https://api.cohere.ai/v1"
    
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
        self.current_key_index = 0
        return True
    
    def is_available(self) -> bool:
        """Check if Cohere handler is properly configured"""
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
            self.models.remove(model_name)
            self.models.insert(0, model_name)
            self.current_model_index = 0
    
    def _supports_system_role(self) -> bool:
        """Cohere supports system roles through preamble"""
        return True
    
    def _build_chat_request(self, prompt: str, messages: List[Dict] = None):
        """Build chat request for Cohere API"""
        processed_messages = self._process_messages(messages or [])
        
        # Extract system message if present
        preamble = ""
        chat_history = []
        current_message = prompt
        
        for msg in processed_messages:
            if msg.get('role') == 'system':
                preamble = msg.get('content', '')
            elif msg.get('role') == 'user':
                chat_history.append({"role": "USER", "message": msg.get('content', '')})
            elif msg.get('role') == 'assistant':
                chat_history.append({"role": "CHATBOT", "message": msg.get('content', '')})
        
        # Check if the prompt is already the last user message to avoid duplication
        if processed_messages and processed_messages[-1].get('content') == prompt:
            # The prompt is already in chat_history, so we need to remove it from history
            # and use it as the current message
            if chat_history and chat_history[-1]["role"] == "USER":
                current_message = chat_history.pop()["message"]
        
        request_data = {
            "message": current_message,
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
        }
        
        if preamble:
            request_data["preamble"] = preamble
        
        if chat_history:
            request_data["chat_history"] = chat_history
        
        return request_data
    
    def generate_stream(self, prompt: str, messages: List[Dict] = None):
        """Generate a streaming response using Cohere API"""
        if not self.is_available():
            yield {
                "success": False,
                "error": "Cohere handler not properly configured - no API keys available",
                "provider": self.name
            }
            return
        
        total_attempts = 0
        models_tried = []
        
        for model_attempt in range(len(self.models)):
            current_model = self.models[self.current_model_index]
            models_tried.append(current_model)
            
            key_attempts = 0
            max_key_attempts = len(self.api_keys)
            
            while key_attempts < max_key_attempts:
                total_attempts += 1
                try:
                    request_data = self._build_chat_request(prompt, messages)
                    request_data["model"] = current_model
                    request_data["stream"] = True
                    
                    url = f"{self.base_url}/chat"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json"
                    }
                    
                    response = requests.post(url, headers=headers, json=request_data, 
                                           timeout=config.API_TIMEOUT, stream=True)
                    
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                line_text = line.decode('utf-8')
                                if line_text.startswith('data: '):
                                    try:
                                        json_data = json.loads(line_text[6:])
                                        if json_data.get('event_type') == 'text-generation':
                                            text = json_data.get('text', '')
                                            if text:
                                                yield {
                                                    "success": True,
                                                    "response": text,
                                                    "provider": self.name,
                                                    "model": current_model,
                                                    "streaming": True
                                                }
                                    except json.JSONDecodeError:
                                        continue
                        return
                    else:
                        logger.warning(f"Cohere {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"Cohere {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All Cohere keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        yield {
            "success": False,
            "error": f"All Cohere models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        }
    
    def generate(self, prompt: str, messages: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response using Cohere API"""
        if not self.is_available():
            return {
                "success": False,
                "error": "Cohere handler not properly configured - no API keys available",
                "provider": self.name
            }
        
        total_attempts = 0
        models_tried = []
        
        for model_attempt in range(len(self.models)):
            current_model = self.models[self.current_model_index]
            models_tried.append(current_model)
            
            key_attempts = 0
            max_key_attempts = len(self.api_keys)
            
            while key_attempts < max_key_attempts:
                total_attempts += 1
                try:
                    request_data = self._build_chat_request(prompt, messages)
                    request_data["model"] = current_model
                    
                    url = f"{self.base_url}/chat"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json"
                    }
                    
                    response = requests.post(url, headers=headers, json=request_data, 
                                           timeout=config.API_TIMEOUT)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        if 'text' in json_data:
                            return {
                                "success": True,
                                "response": json_data['text'],
                                "provider": self.name,
                                "model": current_model,
                                "api_key_index": self.current_key_index,
                                "total_attempts": total_attempts,
                                "models_tried": models_tried,
                                "total_api_keys": len(self.api_keys)
                            }
                        else:
                            logger.warning(f"Cohere {current_model} returned unexpected response format")
                            key_attempts += 1
                            
                            if key_attempts < max_key_attempts:
                                if not self._rotate_api_key():
                                    break
                    else:
                        logger.warning(f"Cohere {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"Cohere {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All Cohere keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        return {
            "success": False,
            "error": f"All Cohere models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        } 