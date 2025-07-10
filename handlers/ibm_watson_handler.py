"""
IBM Watson Handler
Handles API calls to IBM Watson AI services with free tier support
"""

import json
import logging
import requests
from typing import Dict, Any, List
from .base_handler import GenerationHandler
from config import config

logger = logging.getLogger(__name__)


class IBMWatsonHandler(GenerationHandler):
    """Handler for IBM Watson AI services with free tier support"""
    
    def __init__(self):
        self.name = "ibm_watson"
        self.provider_config = config.get_provider('ibm_watson')
        self.models = self.provider_config.models if self.provider_config else []
        self.current_model_index = 0
        self.api_keys = self.provider_config.get_valid_api_keys() if self.provider_config else []
        self.current_key_index = 0
        self.base_url = self.provider_config.base_url if self.provider_config else "https://api.watsonx.ai/v1"
        
        # Watson specific configuration
        self.watson_instance_id = getattr(self.provider_config, 'instance_id', None) if self.provider_config else None
        self.watson_url = getattr(self.provider_config, 'watson_url', None) if self.provider_config else None
    
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
        """Check if IBM Watson handler is properly configured"""
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
        """IBM Watson supports system roles"""
        return True
    
    def _get_access_token(self, api_key: str) -> str:
        """Get access token from IBM Watson"""
        try:
            token_url = "https://iam.cloud.ibm.com/identity/token"
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"
            }
            data = {
                "grant_type": "urn:iam:params:oauth:grant-type:apikey",
                "apikey": api_key
            }
            
            response = requests.post(token_url, headers=headers, data=data, timeout=10)
            if response.status_code == 200:
                return response.json().get('access_token', '')
            else:
                logger.warning(f"Failed to get IBM Watson access token: {response.status_code}")
                return ""
        except Exception as e:
            logger.warning(f"Error getting IBM Watson access token: {str(e)}")
            return ""
    
    def generate_stream(self, prompt: str, messages: List[Dict] = None):
        """Generate a streaming response using IBM Watson API"""
        if not self.is_available():
            yield {
                "success": False,
                "error": "IBM Watson handler not properly configured - no API keys available",
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
                    api_key = self.api_keys[self.current_key_index]
                    access_token = self._get_access_token(api_key)
                    
                    if not access_token:
                        logger.warning(f"IBM Watson key {self.current_key_index + 1} failed to get access token")
                        key_attempts += 1
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                        continue
                    
                    processed_messages = self._process_messages(messages or [])
                    
                    # Build the message string for Watson
                    message_content = ""
                    if processed_messages:
                        for msg in processed_messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if role == 'system':
                                message_content += f"System: {content}\n"
                            elif role == 'user':
                                message_content += f"User: {content}\n"
                            elif role == 'assistant':
                                message_content += f"Assistant: {content}\n"
                    
                    message_content += f"User: {prompt}\nAssistant:"
                    
                    url = f"{self.base_url}/text/generation"
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    }
                    
                    data = {
                        "model_id": current_model,
                        "input": message_content,
                        "parameters": {
                            "max_new_tokens": config.MAX_TOKENS,
                            "temperature": config.TEMPERATURE,
                            "decoding_method": "greedy"
                        }
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
                                        if 'results' in json_data and len(json_data['results']) > 0:
                                            result = json_data['results'][0]
                                            if 'generated_text' in result:
                                                text = result['generated_text']
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
                        logger.warning(f"IBM Watson {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"IBM Watson {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All IBM Watson keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        yield {
            "success": False,
            "error": f"All IBM Watson models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        }
    
    def generate(self, prompt: str, messages: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response using IBM Watson API"""
        if not self.is_available():
            return {
                "success": False,
                "error": "IBM Watson handler not properly configured - no API keys available",
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
                    api_key = self.api_keys[self.current_key_index]
                    access_token = self._get_access_token(api_key)
                    
                    if not access_token:
                        logger.warning(f"IBM Watson key {self.current_key_index + 1} failed to get access token")
                        key_attempts += 1
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                        continue
                    
                    processed_messages = self._process_messages(messages or [])
                    
                    # Build the message string for Watson
                    message_content = ""
                    if processed_messages:
                        for msg in processed_messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if role == 'system':
                                message_content += f"System: {content}\n"
                            elif role == 'user':
                                message_content += f"User: {content}\n"
                            elif role == 'assistant':
                                message_content += f"Assistant: {content}\n"
                    
                    message_content += f"User: {prompt}\nAssistant:"
                    
                    url = f"{self.base_url}/text/generation"
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model_id": current_model,
                        "input": message_content,
                        "parameters": {
                            "max_new_tokens": config.MAX_TOKENS,
                            "temperature": config.TEMPERATURE,
                            "decoding_method": "greedy"
                        }
                    }
                    
                    response = requests.post(url, headers=headers, json=data, 
                                           timeout=config.API_TIMEOUT)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        if 'results' in json_data and len(json_data['results']) > 0:
                            result = json_data['results'][0]
                            if 'generated_text' in result:
                                return {
                                    "success": True,
                                    "response": result['generated_text'],
                                    "provider": self.name,
                                    "model": current_model,
                                    "api_key_index": self.current_key_index,
                                    "total_attempts": total_attempts,
                                    "models_tried": models_tried,
                                    "total_api_keys": len(self.api_keys)
                                }
                        
                        logger.warning(f"IBM Watson {current_model} returned unexpected response format")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    else:
                        logger.warning(f"IBM Watson {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"IBM Watson {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All IBM Watson keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        return {
            "success": False,
            "error": f"All IBM Watson models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        } 