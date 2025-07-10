# 🤖 OnlineAI - Universal AI API Gateway

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Contributors](https://img.shields.io/badge/contributors-welcome-brightgreen.svg)](CONTRIBUTING.md)

A powerful, production-ready Flask API that provides unified access to multiple AI providers with intelligent fallback, load balancing, and streaming support. Features a beautiful web interface for testing and development.

## 🚀 Features

### 🤖 **Multi-Provider AI Support**
- **Google AI (Gemini)** - Latest multimodal models
- **OpenAI** - GPT-4, GPT-3.5, and newer models
- **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
- **DeepSeek** - Affordable coding-focused models
- **Groq** - Ultra-fast inference with Llama/Mixtral
- **Mistral** - European AI with competitive performance
- **xAI (Grok)** - Elon Musk's AI with unique personality
- **Perplexity** - AI search with web access
- **Meta** - Llama 2/3 models
- **Hugging Face** - Access to thousands of open-source models

### 🔧 **Advanced Features**
- 🔄 **Intelligent Fallback System** - Automatically tries different providers if one fails
- 🔑 **Multiple API Key Support** - Load balancing across multiple keys per provider
- 💬 **Conversation Context** - Maintains chat history and context
- 🎯 **Specific Model Selection** - Choose exact models or let the system decide
- 🌊 **Real-time Streaming** - Server-Sent Events (SSE) for live responses
- 🧠 **System Role Support** - Native system instructions with automatic jailbreak fallback
- 🛡️ **Rate Limiting & Caching** - Built-in protection and performance optimization
- 📊 **Performance Monitoring** - Real-time metrics and health checks
- 🎨 **Beautiful Web Interface** - Interactive demo with full feature support

### 🏗️ **Architecture**
- 🔌 **Modular Design** - Easy to add new AI providers
- 🔒 **Secure Configuration** - Environment variables and direct config support
- 📚 **RESTful API** - Clean, well-documented endpoints
- 🐳 **Docker Ready** - Containerized deployment support
- 🧪 **Comprehensive Testing** - Full test coverage with automated testing
- 📖 **Extensive Documentation** - Complete API reference and examples

## 📋 Quick Start

### Prerequisites
- Python 3.7+
- At least one AI provider API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/OnlineAI.git
   cd OnlineAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   
   **Option A: Direct Configuration (Recommended)**
   ```bash
   # Edit config.py and add your API keys
   nano config.py
   ```
   
   **Option B: Environment Variables**
   ```bash
   export GOOGLE_API_KEY=your_google_api_key
   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   # ... add other keys as needed
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the interface:**
   - 🎨 **Web Interface**: http://localhost:5000
   - 📊 **API Documentation**: http://localhost:5000/api
   - 🔧 **Status Dashboard**: http://localhost:5000/status

## 🔑 API Keys & Configuration

### Where to Get API Keys

| Provider | Sign Up Link | Free Tier | Notes |
|----------|-------------|-----------|-------|
| **Google AI** | [AI Studio](https://makersuite.google.com/app/apikey) | ✅ Generous | Best for multimodal tasks |
| **OpenAI** | [Platform](https://platform.openai.com/api-keys) | 💰 Credit | Most reliable, GPT-4 |
| **Anthropic** | [Console](https://console.anthropic.com/) | 💰 Credit | Excellent reasoning |
| **DeepSeek** | [Platform](https://platform.deepseek.com/) | 💰 Very cheap | Great for coding |
| **Groq** | [Console](https://console.groq.com/) | ✅ Limited | Ultra-fast inference |
| **Mistral** | [Platform](https://console.mistral.ai/) | 💰 Credit | European AI |
| **xAI** | [Console](https://console.x.ai/) | 💰 Credit | Unique personality |
| **Perplexity** | [Pro](https://www.perplexity.ai/) | 💰 Credit | Web search AI |
| **Meta** | [Llama API](https://www.llama-api.com/) | 💰 Credit | Llama models |
| **Hugging Face** | [Tokens](https://huggingface.co/settings/tokens) | ✅ Limited | Open source models |

### Configuration Examples

**Multiple Keys for Load Balancing:**
```python
# config.py
GOOGLE_API_KEYS = [
    "AIzaSyC_key1_...",
    "AIzaSyC_key2_...",
    "AIzaSyC_key3_..."
]

OPENAI_API_KEYS = [
    "sk-key1_...",
    "sk-key2_..."
]
```

**Provider-Specific Settings:**
```python
# config.py
DEFAULT_AI_PROVIDER = "google"  # First provider to try
MAX_TOKENS = 2048              # Max response length
TEMPERATURE = 0.7              # Creativity level (0.0-1.0)
API_TIMEOUT = 45               # Request timeout in seconds
```

## 📚 API Reference

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 🏠 **GET /** - Home Page
Returns the interactive web interface.

#### 📊 **GET /api** - API Documentation
Returns comprehensive API documentation.

#### 🔧 **GET /status** - Status Check
Returns status of all configured providers.

**Response:**
```json
{
  "total_handlers": 10,
  "available_handlers": 7,
  "handlers": [
    {
      "name": "google",
      "available": true,
      "streaming_support": true,
      "details": "3 API keys configured"
    }
  ]
}
```

#### 🏥 **GET /health** - Health Check
Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

#### 🤖 **POST /generate** - Generate AI Response

**Request Body:**
```json
{
  "prompt": "Write a Python function to calculate fibonacci numbers",
  "preference": "openai:gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful programming assistant"
    },
    {
      "role": "user", 
      "content": "Write clean, documented code"
    }
  ],
  "stream": false,
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Parameters:**
- `prompt` (string, required): The main prompt/question
- `preference` (string, optional): Preferred provider or model (e.g., "google", "openai:gpt-4")
- `messages` (array, optional): Conversation history
- `stream` (boolean, optional): Enable streaming response (default: false)
- `max_tokens` (integer, optional): Maximum response length
- `temperature` (float, optional): Response creativity (0.0-1.0)

**Non-Streaming Response:**
```json
{
  "success": true,
  "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "provider": "openai",
  "model": "gpt-4",
  "api_key_index": 0,
  "total_api_keys": 2,
  "models_tried": ["gpt-4"],
  "total_attempts": 1,
  "processing_time": 2.34,
  "fallback_info": {
    "handlers_tried": ["openai"],
    "total_handlers_available": 7,
    "handler_used": "openai"
  }
}
```

**Streaming Response:**
```
Content-Type: text/event-stream

data: {"success": true, "processing": true, "message": "Starting generation..."}
data: {"success": true, "processing": true, "message": "Trying openai handler..."}
data: {"success": true, "response": "def fibonacci(n):", "streaming": true, "provider": "openai"}
data: {"success": true, "response": "\n    if n <= 1:", "streaming": true, "provider": "openai"}
data: {"success": true, "done": true, "final_response": "Complete response here"}
```

**Error Response:**
```json
{
  "success": false,
  "error": "All AI generation handlers failed",
  "handlers_tried": ["openai", "google", "anthropic"],
  "total_handlers": 3,
  "errors": [
    {
      "handler": "openai",
      "error": "Rate limit exceeded"
    },
    {
      "handler": "google", 
      "error": "API key invalid"
    }
  ]
}
```

## 🛠️ Usage Examples

### Python Client
```python
import requests
import json

# Basic request
response = requests.post('http://localhost:5000/generate', 
    json={
        'prompt': 'Explain machine learning in simple terms',
        'preference': 'google',
        'max_tokens': 500
    })

result = response.json()
print(result['response'])
```

### cURL
```bash
# Basic request
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about coding",
    "preference": "anthropic:claude-3-sonnet",
    "temperature": 0.9
  }'

# Streaming request
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a story",
    "stream": true
  }'
```

### JavaScript/Node.js
```javascript
// Non-streaming
const response = await fetch('http://localhost:5000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'Explain quantum computing',
    preference: 'openai:gpt-4',
    max_tokens: 1000
  })
});

const result = await response.json();
console.log(result.response);

// Streaming with EventSource
const eventSource = new EventSource('http://localhost:5000/generate', {
  method: 'POST',
  body: JSON.stringify({
    prompt: 'Write a poem',
    stream: true
  })
});

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.response) {
    console.log(data.response);
  }
};
```

## 🧪 Testing

### Run the Test Suite
```bash
# Run all tests
python test_onlineai.py

# Run specific test categories
python test_onlineai.py --providers google openai
python test_onlineai.py --endpoints generate status
python test_onlineai.py --features streaming fallback
```

### Manual Testing
```bash
# Test basic functionality
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'

# Test provider status
curl http://localhost:5000/status

# Test health check
curl http://localhost:5000/health
```

## 🏗️ Architecture Overview

```
OnlineAI/
├── app.py                    # Main Flask application
├── config.py                 # Configuration management
├── handlers/                 # AI provider handlers
│   ├── __init__.py
│   ├── base_handler.py      # Base handler interface
│   ├── google_handler.py    # Google AI implementation
│   ├── openai_handler.py    # OpenAI implementation
│   ├── anthropic_handler.py # Anthropic implementation
│   └── ...                  # Other provider handlers
├── tests/                   # Test suite
│   └── test_onlineai.py    # Comprehensive test script
├── index.html              # Web interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Key Components

1. **AiHandler** - Main orchestrator that manages fallback logic
2. **GenerationHandler** - Base class for all AI providers
3. **Provider Handlers** - Individual implementations for each AI service
4. **Configuration System** - Flexible config management with environment fallback
5. **Web Interface** - Interactive testing and demonstration interface

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Core Settings
export FLASK_DEBUG=true
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000

# AI Settings
export DEFAULT_AI_PROVIDER=google
export MAX_TOKENS=2048
export TEMPERATURE=0.7
export API_TIMEOUT=45

# Performance
export MAX_CONCURRENT_REQUESTS=10
export RATE_LIMIT_PER_MINUTE=100
export ENABLE_CACHING=true
export CACHE_TTL=3600

# Logging
export LOG_LEVEL=INFO
export LOG_REQUESTS=true
export LOG_PERFORMANCE=true
```

### Custom Provider Configuration
```python
# config.py - Add custom provider
class CustomProviderConfig(ProviderConfig):
    def __init__(self):
        super().__init__(
            name='custom',
            api_keys=os.getenv('CUSTOM_API_KEYS', '').split(','),
            models=['custom-model-1', 'custom-model-2'],
            base_url='https://api.custom.com/v1',
            supports_streaming=True,
            max_context_tokens=32000
        )
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

```bash
# Build and run
docker build -t onlineai .
docker run -p 5000:5000 -e GOOGLE_API_KEY=your_key onlineai
```

### Production Settings
```python
# config.py - Production configuration
FLASK_DEBUG = False
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
API_TIMEOUT = 30
MAX_CONCURRENT_REQUESTS = 50
RATE_LIMIT_PER_MINUTE = 1000
ENABLE_PERFORMANCE_TRACKING = True
LOG_LEVEL = "WARNING"
```

## 📈 Performance Optimization

### Caching
```python
# Enable response caching
ENABLE_CACHING = True
CACHE_TTL = 3600  # 1 hour

# Redis cache (optional)
REDIS_URL = "redis://localhost:6379"
```

### Load Balancing
```python
# Multiple API keys for load balancing
GOOGLE_API_KEYS = [
    "key1", "key2", "key3"
]

# Automatic rotation and failover
ENABLE_KEY_ROTATION = True
```

## 🔍 Troubleshooting

### Common Issues

**1. No API Keys Configured**
```
Error: No AI generation handlers are properly configured
Solution: Add at least one valid API key to config.py
```

**2. Rate Limiting**
```
Error: Rate limit exceeded
Solution: Add multiple API keys or reduce request frequency
```

**3. Model Not Found**
```
Error: Model 'gpt-5' not found
Solution: Check available models in /status endpoint
```

**4. Streaming Not Working**
```
Error: Streaming not supported
Solution: Ensure client supports Server-Sent Events (SSE)
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export FLASK_DEBUG=true

# Run with detailed output
python app.py
```

### Health Checks
```bash
# Check all providers
curl http://localhost:5000/status

# Check specific provider
curl "http://localhost:5000/status?provider=openai"

# Test generation
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "preference": "google"}'
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repo
git clone https://github.com/yourusername/OnlineAI.git
cd OnlineAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_onlineai.py

# Start development server
python app.py
```

### Adding New Providers
1. Create handler in `handlers/new_provider_handler.py`
2. Extend `GenerationHandler` base class
3. Add configuration in `config.py`
4. Update `handlers/__init__.py`
5. Add tests in `test_onlineai.py`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all AI providers for their excellent APIs
- Built with Flask and modern Python practices
- Inspired by the need for reliable AI infrastructure

## 📞 Support

- 📧 Email: ethachu21@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/ethachu21/OnlineAI/issues)

---

**Made with ❤️ by ethachu21** 
