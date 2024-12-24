# Ollama Integration Architecture

## Understanding Ollama vs langchain-ollama

### Local Ollama Installation
- Ollama is installed directly on your machine as a service
- Runs on `localhost:11434` by default
- Manages the actual LLM models (like llama3.2:3b)
- Handles model loading, inference, and API endpoints

### Python Package (langchain-ollama)
- Installed in your Python virtual environment
- Acts as a client library to communicate with local Ollama service
- Provides Python interfaces and abstractions through LangChain
- Does NOT contain any actual LLM models

## Architecture Diagram
```
Your Application (Python)
        ↓
   langchain-ollama
   (Python Package)
        ↓
   HTTP Requests
        ↓
   Ollama Service
  (Local Installation)
        ↓
    LLM Models
```

## Why Both Are Needed

1. **Separation of Concerns**
   - Ollama: Handles model execution and serving
   - langchain-ollama: Handles Python integration and communication

2. **Analogy**
   - Ollama is like a database server
   - langchain-ollama is like a database client library
   - You need both to build a database application

3. **Benefits**
   - Clean API interface through LangChain
   - Type safety and Python integration
   - Error handling and retries
   - Integration with other LangChain features

## Version Compatibility

### Model Versioning
- The model version is controlled by your local Ollama installation
- langchain-ollama simply requests models by name
- Example:
  ```python
  from langchain_ollama import OllamaLLM
  
  # This uses whatever version of llama3.2:3b is installed in your local Ollama
  ollama = OllamaLLM(
      base_url="http://localhost:11434",
      model="llama3.2:3b"
  )
  ```

### Checking Available Models
```bash
# List models installed in local Ollama
ollama list

# Pull specific model version
ollama pull llama3.2:3b

# Run model to verify it works
ollama run llama3.2:3b "Hello, are you working?"
```

### Ensuring Model Compatibility
1. **Check Local Installation**
   ```bash
   # Verify Ollama is running
   curl http://localhost:11434/api/version
   
   # List available models
   ollama list
   ```

2. **Verify Model in Code**
   ```python
   import requests
   
   def verify_model(model_name="llama3.2:3b"):
       try:
           # Check if model exists
           response = requests.post(
               "http://localhost:11434/api/generate",
               json={"model": model_name, "prompt": "test"}
           )
           if response.status_code == 200:
               return True
           return False
       except Exception:
           return False
   ```

## Best Practices

1. **Model Management**
   - Always verify model availability in local Ollama before using
   - Use exact model names as shown in `ollama list`
   - Keep local Ollama models updated
   - Test model responses before production use

2. **Configuration**
   - Use environment variables for Ollama host/port
   - Log model usage and versions
   - Handle connection errors gracefully
   - Set appropriate timeouts for API calls

3. **Testing**
   - Verify Ollama connection before operations
   - Include fallback mechanisms
   - Monitor model performance
   - Log all API interactions for debugging

## Example Implementation
```python
import os
import requests
import logging
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class OllamaIntegration:
    def __init__(self, model_name="llama3.2:3b"):
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.model_name = model_name
        
        # Initialize client
        self.verify_connection()
        self.ollama = OllamaLLM(
            base_url=f'http://{self.ollama_host}:{self.ollama_port}',
            model=self.model_name,
            temperature=0.7
        )

    def verify_connection(self):
        """Verify Ollama connection and model availability"""
        try:
            # Check Ollama service
            url = f'http://{self.ollama_host}:{self.ollama_port}/api/version'
            response = requests.get(url)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to connect to Ollama: {response.text}")
            
            # Verify model exists
            response = requests.post(
                f'http://{self.ollama_host}:{self.ollama_port}/api/generate',
                json={"model": self.model_name, "prompt": "test"}
            )
            if response.status_code != 200:
                raise ValueError(f"Model {self.model_name} not available")
            
            logger.info(f"Successfully connected to Ollama with model {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying Ollama setup: {e}")
            raise
```

## Troubleshooting

1. **Connection Issues**
   - Verify Ollama is running: `curl http://localhost:11434/api/version`
   - Check if model is downloaded: `ollama list`
   - Try running model directly: `ollama run llama3.2:3b "test"`

2. **Model Problems**
   - Pull model again: `ollama pull llama3.2:3b`
   - Check model compatibility
   - Verify model name matches exactly

3. **API Errors**
   - Check logs for detailed error messages
   - Verify network connectivity
   - Ensure correct model name in code