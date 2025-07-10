"""
OpenRouter Handler
Handles API calls to OpenRouter which provides access to multiple AI models
"""

import json
import logging
import requests
from typing import Dict, Any, List
from .base_handler import GenerationHandler
from config import config

logger = logging.getLogger(__name__)


class OpenRouterHandler(GenerationHandler):
    """Handler for OpenRouter API with access to multiple models"""
    
    def __init__(self):
        self.name = "openrouter"
        self.provider_config = config.get_provider('openrouter')
        self.models = self.provider_config.models if self.provider_config else []
        self.current_model_index = 0
        self.api_keys = self.provider_config.get_valid_api_keys() if self.provider_config else []
        self.current_key_index = 0
        self.base_url = self.provider_config.base_url if self.provider_config else "https://openrouter.ai/api/v1"
    
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
        """Check if OpenRouter handler is properly configured"""
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
        """OpenRouter supports system roles like OpenAI"""
        return True
    
    def generate_stream(self, prompt: str, messages: List[Dict] = None):
        """Generate a streaming response using OpenRouter API"""
        if not self.is_available():
            yield {
                "success": False,
                "error": "OpenRouter handler not properly configured - no API keys available",
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
                    processed_messages = self._process_messages(messages or [])
                    
                    openrouter_messages = []
                    if processed_messages:
                        for msg in processed_messages:
                            openrouter_messages.append({
                                "role": msg.get('role', 'user'),
                                "content": msg.get('content', '')
                            })
                    
                    # Only add the current prompt if it's not already the last user message
                    if not processed_messages or processed_messages[-1].get('content') != prompt:
                        openrouter_messages.append({
                            "role": "user",
                            "content": prompt
                        })
                    
                    url = f"{self.base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/your-repo/onlineai",  # Required by OpenRouter
                        "X-Title": "OnlineAI"  # Optional but recommended
                    }
                    
                    data = {
                        "model": current_model,
                        "messages": openrouter_messages,
                        "max_tokens": config.MAX_TOKENS,
                        "temperature": config.TEMPERATURE,
                        "stream": True
                    }
                    
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
                        logger.warning(f"OpenRouter {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"OpenRouter {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All OpenRouter keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        yield {
            "success": False,
            "error": f"All OpenRouter models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        }
    
    def generate(self, prompt: str, messages: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response using OpenRouter API"""
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenRouter handler not properly configured - no API keys available",
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
                    processed_messages = self._process_messages(messages or [])
                    
                    openrouter_messages = []
                    if processed_messages:
                        for msg in processed_messages:
                            openrouter_messages.append({
                                "role": msg.get('role', 'user'),
                                "content": msg.get('content', '')
                            })
                    
                    # Only add the current prompt if it's not already the last user message
                    if not processed_messages or processed_messages[-1].get('content') != prompt:
                        openrouter_messages.append({
                            "role": "user",
                            "content": prompt
                        })
                    
                    url = f"{self.base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/your-repo/onlineai",
                        "X-Title": "OnlineAI"
                    }
                    
                    data = {
                        "model": current_model,
                        "messages": openrouter_messages,
                        "max_tokens": config.MAX_TOKENS,
                        "temperature": config.TEMPERATURE
                    }
                    
                    response = requests.post(url, headers=headers, json=data, 
                                           timeout=config.API_TIMEOUT)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            choice = json_data['choices'][0]
                            if 'message' in choice and 'content' in choice['message']:
                                return {
                                    "success": True,
                                    "response": choice['message']['content'],
                                    "provider": self.name,
                                    "model": current_model,
                                    "api_key_index": self.current_key_index,
                                    "total_attempts": total_attempts,
                                    "models_tried": models_tried,
                                    "total_api_keys": len(self.api_keys)
                                }
                        
                        logger.warning(f"OpenRouter {current_model} returned unexpected response format")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    else:
                        logger.warning(f"OpenRouter {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"OpenRouter {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All OpenRouter keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        return {
            "success": False,
            "error": f"All OpenRouter models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        } 