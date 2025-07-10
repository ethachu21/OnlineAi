"""
Hugging Face Handler
Handles API calls to Hugging Face models
"""

import json
import logging
import requests
from typing import Dict, Any, List
from .base_handler import GenerationHandler
from config import config

logger = logging.getLogger(__name__)


class HuggingFaceHandler(GenerationHandler):
    """Handler for Hugging Face models with multiple model support"""
    
    def __init__(self):
        self.name = "huggingface"
        self.provider_config = config.get_provider('huggingface')
        self.models = self.provider_config.models if self.provider_config else []
        self.current_model_index = 0
        self.api_keys = self.provider_config.get_valid_api_keys() if self.provider_config else []
        self.current_key_index = 0
        self.base_url = self.provider_config.base_url if self.provider_config else "https://api-inference.huggingface.co/models"
    
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
        """Check if Hugging Face handler is properly configured"""
        return len(self.api_keys) > 0 and self.provider_config is not None
    
    def get_name(self) -> str:
        """Get the name of this handler"""
        return self.name
    
    def supports_streaming(self) -> bool:
        """Check if this handler supports streaming"""
        return False  # Hugging Face Inference API doesn't support streaming
    
    def set_preferred_model(self, model_name: str):
        """Set a specific model as preferred for this handler"""
        if model_name in self.models:
            # Move the preferred model to the front
            self.models.remove(model_name)
            self.models.insert(0, model_name)
            self.current_model_index = 0
    
    def _supports_system_role(self) -> bool:
        """Hugging Face models don't support system roles"""
        return False  # Will use jailbreak method
    
    def generate_stream(self, prompt: str, messages: List[Dict] = None):
        """Generate a streaming response (not supported by Hugging Face)"""
        # Hugging Face doesn't support streaming, so we'll fall back to regular generation
        result = self.generate(prompt, messages)
        if result.get("success"):
            yield {
                "success": True,
                "response": result["response"],
                "provider": self.name,
                "model": result.get("model"),
                "streaming": False
            }
        else:
            yield result
    
    def generate(self, prompt: str, messages: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response using Hugging Face API with model and key rotation"""
        if not self.is_available():
            return {
                "success": False,
                "error": "Hugging Face handler not properly configured - no API keys available",
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
                    
                    # Build the input text for Hugging Face
                    input_text = ""
                    
                    if processed_messages:
                        for msg in processed_messages:
                            if msg.get('role') == 'system':
                                # Use jailbreak technique for system messages
                                input_text += f"[SYSTEM]: {msg.get('content', '')}\n"
                            elif msg.get('role') == 'user':
                                input_text += f"User: {msg.get('content', '')}\n"
                            elif msg.get('role') == 'assistant':
                                input_text += f"Assistant: {msg.get('content', '')}\n"
                    
                    # Only add the current prompt if it's not already the last user message
                    if not processed_messages or processed_messages[-1].get('content') != prompt:
                        input_text += f"User: {prompt}\nAssistant:"
                    else:
                        input_text += "Assistant:"
                    
                    # Prepare the request
                    url = f"{self.base_url}/{current_model}"
                    headers = {
                        "Authorization": f"Bearer {self.api_keys[self.current_key_index]}",
                        "Content-Type": "application/json"
                    }
                    
                    # Different models may have different input formats
                    if "DialoGPT" in current_model or "blenderbot" in current_model:
                        # For conversational models, use inputs parameter
                        data = {
                            "inputs": input_text,
                            "parameters": {
                                "max_length": min(config.MAX_TOKENS, 1024),  # HF models often have smaller limits
                                "temperature": config.TEMPERATURE,
                                "do_sample": True,
                                "return_full_text": False
                            }
                        }
                    else:
                        # For other models, use a simpler format
                        data = {
                            "inputs": input_text,
                            "parameters": {
                                "max_new_tokens": min(config.MAX_TOKENS, 512),
                                "temperature": config.TEMPERATURE,
                                "return_full_text": False
                            }
                        }
                    
                    # Make the API call
                    response = requests.post(url, headers=headers, json=data, 
                                           timeout=config.API_TIMEOUT)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        # Handle different response formats
                        content = None
                        
                        if isinstance(json_data, list) and len(json_data) > 0:
                            # Most HF models return a list
                            first_result = json_data[0]
                            if isinstance(first_result, dict):
                                if 'generated_text' in first_result:
                                    content = first_result['generated_text']
                                elif 'text' in first_result:
                                    content = first_result['text']
                        elif isinstance(json_data, dict):
                            # Some models return a dict
                            if 'generated_text' in json_data:
                                content = json_data['generated_text']
                            elif 'text' in json_data:
                                content = json_data['text']
                        
                        if content:
                            # Clean up the response
                            content = content.strip()
                            if content.startswith(input_text):
                                content = content[len(input_text):].strip()
                            
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
                            "error": "No valid response from Hugging Face",
                            "provider": self.name,
                            "model": current_model,
                            "response_data": json_data
                        }
                    elif response.status_code == 503:
                        # Model is loading, this is common with HF
                        logger.warning(f"Hugging Face {current_model} is loading, trying next key/model")
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    else:
                        logger.warning(f"Hugging Face {current_model} key {self.current_key_index + 1} failed: HTTP {response.status_code}")
                        try:
                            error_data = response.json()
                            logger.warning(f"Error details: {error_data}")
                        except:
                            pass
                        
                        key_attempts += 1
                        
                        if key_attempts < max_key_attempts:
                            if not self._rotate_api_key():
                                break
                    
                except Exception as e:
                    logger.warning(f"Hugging Face {current_model} key {self.current_key_index + 1} failed: {str(e)}")
                    key_attempts += 1
                    
                    if key_attempts < max_key_attempts:
                        if not self._rotate_api_key():
                            break
            
            # All keys failed for this model, try next model
            if model_attempt < len(self.models) - 1:
                if self._rotate_model():
                    logger.info(f"All Hugging Face keys failed for {current_model}, trying {self.models[self.current_model_index]}")
        
        # All models and keys failed
        return {
            "success": False,
            "error": f"All Hugging Face models and keys failed. Models tried: {models_tried}",
            "provider": self.name,
            "models_tried": models_tried,
            "total_attempts": total_attempts,
            "total_api_keys": len(self.api_keys)
        } 