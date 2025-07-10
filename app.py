import os
import json
import logging
import asyncio
from flask import Flask, jsonify, request, send_from_directory
from typing import Dict, Any, List

# Import configuration
try:
    from config import config
except ImportError:
    print("❌ config.py not found. Please create config.py with your API keys.")
    print("💡 You can copy and rename config.py.example to config.py")
    exit(1)

# Import handlers
try:
    from handlers.openai_handler import OpenAiHandler
    from handlers.anthropic_handler import AnthropicHandler
    from handlers.google_handler import GoogleAiHandler
    from handlers.groq_handler import GroqHandler
    from handlers.mistral_handler import MistralHandler
    from handlers.perplexity_handler import PerplexityHandler
    from handlers.huggingface_handler import HuggingFaceHandler
    from handlers.deepseek_handler import DeepSeekHandler
    from handlers.meta_handler import MetaHandler
    from handlers.xai_handler import XaiHandler
    from handlers.cohere_handler import CohereHandler
    from handlers.openrouter_handler import OpenRouterHandler
    from handlers.ibm_watson_handler import IBMWatsonHandler
except ImportError as e:
    print(f"❌ Error importing handlers: {e}")
    print("💡 Please ensure all handler files are created in the handlers/ directory")
    exit(1)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AiHandler:
    """Main AI handler that manages multiple generation methods with fallback"""
    
    def __init__(self, prompt: str, preference: str = None, messages: List[Dict] = None):
        self.prompt = prompt
        self.preference = preference or config.DEFAULT_AI_PROVIDER
        self.messages = messages or []
        
        # Initialize all available handlers
        self.handlers = self._initialize_handlers()
        
        # Validate that at least one handler is available
        available_handlers = [h for h in self.handlers if h.is_available()]
        if not available_handlers:
            raise ValueError("No AI generation handlers are properly configured")
    
    def _initialize_handlers(self) -> List:
        """Initialize only handlers for providers with configured API keys"""
        handlers = []
        
        # Only add handlers for providers that have API keys configured
        if config.is_provider_enabled("google"):
            handlers.append(GoogleAiHandler())
        if config.is_provider_enabled("openai"):
            handlers.append(OpenAiHandler())
        if config.is_provider_enabled("anthropic"):
            handlers.append(AnthropicHandler())
        if config.is_provider_enabled("deepseek"):
            handlers.append(DeepSeekHandler())
        if config.is_provider_enabled("groq"):
            handlers.append(GroqHandler())
        if config.is_provider_enabled("mistral"):
            handlers.append(MistralHandler())
        if config.is_provider_enabled("xai"):
            handlers.append(XaiHandler())
        if config.is_provider_enabled("perplexity"):
            handlers.append(PerplexityHandler())
        if config.is_provider_enabled("meta"):
            handlers.append(MetaHandler())
        if config.is_provider_enabled("huggingface"):
            handlers.append(HuggingFaceHandler())
        if config.is_provider_enabled("cohere"):
            handlers.append(CohereHandler())
        if config.is_provider_enabled("openrouter"):
            handlers.append(OpenRouterHandler())
        if config.is_provider_enabled("ibm_watson"):
            handlers.append(IBMWatsonHandler())
        
        enabled_providers = [h.get_name() for h in handlers]
        logger.info(f"Initialized handlers for providers: {enabled_providers}")
        
        return handlers
    
    def _get_ordered_handlers(self) -> List:
        """Get handlers in the order they should be tried"""
        available_handlers = [h for h in self.handlers if h.is_available()]
        
        if not available_handlers:
            return []
        
        # If a preference is specified, try to put it first
        if self.preference:
            # Check if preference is a specific model (contains provider:model format)
            if ':' in self.preference:
                provider_name, model_name = self.preference.split(':', 1)
                preferred_handlers = [h for h in available_handlers if h.get_name() == provider_name]
                other_handlers = [h for h in available_handlers if h.get_name() != provider_name]
                
                # Set the specific model for the preferred handler
                if preferred_handlers:
                    preferred_handlers[0].set_preferred_model(model_name)
                    
                return preferred_handlers + other_handlers
            else:
                # Regular provider preference
                preferred_handlers = [h for h in available_handlers if h.get_name() == self.preference]
                other_handlers = [h for h in available_handlers if h.get_name() != self.preference]
                return preferred_handlers + other_handlers
        
        # Default order: return all available handlers
        return available_handlers
    
    async def generate(self) -> Dict[str, Any]:
        """Generate a response using the fallback system"""
        ordered_handlers = self._get_ordered_handlers()
        
        if not ordered_handlers:
            return {
                "success": False,
                "error": "No AI generation handlers are available",
                "handlers_tried": [],
                "total_handlers": len(self.handlers)
            }
        
        errors = []
        handlers_tried = []
        
        for handler in ordered_handlers:
            handler_name = handler.get_name()
            handlers_tried.append(handler_name)
            
            logger.info(f"Trying {handler_name} handler...")
            
            try:
                # Run handler generate in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler.generate, self.prompt, self.messages)
                
                if result.get("success"):
                    # Success! Add metadata about the fallback process
                    result["fallback_info"] = {
                        "handlers_tried": handlers_tried,
                        "total_handlers_available": len(ordered_handlers),
                        "handler_used": handler_name
                    }
                    
                    return result
                else:
                    # Handler failed, add to errors and try next
                    errors.append({
                        "handler": handler_name,
                        "error": result.get("error", "Unknown error")
                    })
                    logger.warning(f"{handler_name} handler failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                # Unexpected error with handler
                error_msg = f"Unexpected error with {handler_name} handler: {str(e)}"
                errors.append({
                    "handler": handler_name,
                    "error": error_msg
                })
                logger.error(error_msg)
        
        # All handlers failed
        return {
            "success": False,
            "error": "All AI generation handlers failed",
            "handlers_tried": handlers_tried,
            "total_handlers": len(ordered_handlers),
            "errors": errors
        }
    
    def generate_stream(self):
        """Generate a streaming response using the fallback system"""
        ordered_handlers = self._get_ordered_handlers()
        
        if not ordered_handlers:
            yield {
                "success": False,
                "error": "No AI generation handlers are available",
                "handlers_tried": [],
                "total_handlers": len(self.handlers)
            }
            return
        
        errors = []
        handlers_tried = []
        
        for handler in ordered_handlers:
            handler_name = handler.get_name()
            handlers_tried.append(handler_name)
            
            logger.info(f"Trying {handler_name} handler for streaming...")
            
            try:
                # Check if handler supports streaming
                if handler.supports_streaming():
                    # Try streaming generation - don't use run_in_executor for generators
                    stream_gen = handler.generate_stream(self.prompt, self.messages)
                    
                    # Stream the results
                    chunks_received = 0
                    for chunk in stream_gen:
                        if chunk.get("success"):
                            chunk["fallback_info"] = {
                                "handlers_tried": handlers_tried,
                                "total_handlers_available": len(ordered_handlers),
                                "handler_used": handler_name
                            }
                            yield chunk
                            chunks_received += 1
                            
                            # If streaming is complete, return
                            if chunk.get("done"):
                                return
                        else:
                            # Stream failed, break to try next handler
                            logger.warning(f"{handler_name} streaming failed: {chunk.get('error', 'Unknown error')}")
                            break
                    else:
                        # Stream completed successfully (no break occurred)
                        if chunks_received > 0:
                            return
                        else:
                            # No chunks received, try next handler
                            logger.warning(f"{handler_name} streaming produced no chunks")
                            
                else:
                    # Handler doesn't support streaming, fallback to regular generation
                    logger.info(f"{handler_name} doesn't support streaming, using regular generation")
                    
                    result = handler.generate(self.prompt, self.messages)
                    
                    if result.get("success"):
                        # Success! Add metadata about the fallback process and stream the full response
                        result["fallback_info"] = {
                            "handlers_tried": handlers_tried,
                            "total_handlers_available": len(ordered_handlers),
                            "handler_used": handler_name
                        }
                        result["streaming"] = False  # Indicate this was not true streaming
                        yield result
                        return
                    else:
                        # Handler failed, add to errors and try next
                        errors.append({
                            "handler": handler_name,
                            "error": result.get("error", "Unknown error")
                        })
                        logger.warning(f"{handler_name} handler failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                # Unexpected error with handler
                error_msg = f"Unexpected error with {handler_name} handler: {str(e)}"
                errors.append({
                    "handler": handler_name,
                    "error": error_msg
                })
                logger.error(error_msg)
        
        # All handlers failed
        yield {
            "success": False,
            "error": "All AI generation handlers failed",
            "handlers_tried": handlers_tried,
            "total_handlers": len(ordered_handlers),
            "errors": errors
        }


# Create Flask app
app = Flask(__name__)

@app.before_request
def log_request():
    """Log incoming requests if enabled"""
    if config.LOG_REQUESTS:
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.route('/')
async def home():
    """Home endpoint - serves the demo page"""
    return send_from_directory('.', 'index.html')

@app.route('/api')
async def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "Welcome to the AI Handler API with Fallback System!",
        "version": "3.0",
        "features": {
            "streaming": True,
            "fallback_system": True,
            "multi_provider": True,
            "api_key_rotation": True,
            "class_based_config": True,
            "modular_handlers": True,
            "direct_config": True,
            "interactive_setup": True,
            "async_support": True
        },
        "endpoints": {
            "/": "Demo page",
            "/api": "API information",
            "/generate": "Generate AI response (streaming or non-streaming)",
            "/status": "Check handler status",
            "/health": "Health check"
        },
        "configuration": config.to_dict()
    })

@app.route('/generate', methods=['POST'])
async def generate():
    """Generate AI response with optional streaming support"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    prompt = data.get('prompt', '')
    preference = data.get('preference')
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    
    # If prompt is empty but we have messages, use the last user message as the prompt
    if not prompt and messages:
        # Find the last user message to use as the prompt
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                prompt = msg.get('content', '')
                break
    
    if not prompt:
        return jsonify({"error": "Prompt is required (either as 'prompt' field or as a user message)"}), 400
    
    try:
        # Create AiHandler instance
        handler = AiHandler(prompt, preference, messages)
        
        if stream:
            # Return streaming response
            def generate_response():
                try:
                    # Send initial processing message
                    yield f"data: {json.dumps({'success': True, 'processing': True, 'message': 'Starting AI generation...'})}\n\n"
                    
                    logger.info("Starting streaming generation...")
                    chunk_count = 0
                    for chunk in handler.generate_stream():
                        chunk_count += 1
                        logger.info(f"Yielding chunk {chunk_count}: {chunk}")
                        # Format as Server-Sent Events (SSE)
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    logger.info(f"Streaming completed with {chunk_count} chunks")
                    # Send final event to indicate completion
                    yield f"data: {json.dumps({'success': True, 'done': True})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"data: {json.dumps({'success': False, 'error': str(e)})}\n\n"
            
            return app.response_class(
                generate_response(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        else:
            # Return regular JSON response
            result = await handler.generate()
            
            if result.get('success'):
                return jsonify(result)
            else:
                return jsonify(result), 500
                
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
async def status():
    """Get status of all available AI handlers"""
    try:
        # Create a temporary handler to check status
        temp_handler = AiHandler("test", None, [])
        
        handler_status = []
        for handler in temp_handler.handlers:
            handler_status.append({
                "name": handler.get_name(),
                "available": handler.is_available(),
                "streaming_support": handler.supports_streaming(),
                "details": "Configured" if handler.is_available() else "Not configured"
            })
        
        available_count = sum(1 for h in temp_handler.handlers if h.is_available())
        
        return jsonify({
            "total_handlers": len(temp_handler.handlers),
            "available_handlers": available_count,
            "handlers": handler_status,
            "configuration": config.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({"error": f"Error checking status: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
async def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": os.getpid(),
        "version": "3.0",
        "async": True
    })

def main():
    """Main entry point for the application"""
    # Get Flask configuration from config.py
    debug = config.FLASK_DEBUG
    host = config.FLASK_HOST
    port = config.FLASK_PORT
    
    logger.info(f"Starting async Flask app on {host}:{port} with debug={debug}")
    
    # Show enabled providers on startup
    enabled_providers = config.get_enabled_providers()
    if enabled_providers:
        logger.info(f"Enabled AI providers: {enabled_providers}")
    else:
        logger.warning("No AI providers enabled! Please configure API keys in config.py")
    
    app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    main()
