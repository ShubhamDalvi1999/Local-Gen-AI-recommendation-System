# Classy Books Web - Project Overview

## Project Description
Classy Books Web is an intelligent book recommendation system that combines traditional machine learning with modern NLP techniques and Large Language Models (LLM) to provide personalized book recommendations.

## Core Components

### 1. Web Interface
- Flask-based web application
- Interactive chat interface
- Database management portal
- Form-based recommendation system

### 2. Recommendation Systems

#### Traditional ML Pipeline
```
User Form Input → ML Model → Book Prediction
```
- Uses classifier.pickle for traditional ML predictions
- Form-based questionnaire for user preferences
- Direct book recommendations based on classification

#### NLP-Enhanced Chatbot
```
User Query → Vector Embeddings → Similarity Search → LLM Response
```
- BERT embeddings for semantic understanding
- FAISS vector store for efficient similarity search
- Ollama LLM for natural language interaction
- Contextual recommendations with explanations

### 3. Database Architecture
```
SQLite Database
├── Books Table
│   ├── id (INTEGER PRIMARY KEY)
│   ├── book (TEXT)
│   ├── link (TEXT)
│   ├── vector (BLOB)
│   └── description (TEXT)
└── Books_backup (Migration Backup)
```

## Technical Stack

### Core Technologies
- Python 3.8+
- Flask 3.0.3
- SQLite3
- FAISS (Facebook AI Similarity Search)
- Ollama LLM (llama3.2:3b model)

### Key Libraries
```
Machine Learning & NLP
├── sentence-transformers (BERT)
├── scikit-learn
├── numpy/scipy
├── FAISS-cpu
└── langchain-ollama

Web Framework
├── Flask
├── Werkzeug
└── Jinja2

Utilities
├── requests
├── logging
└── JSON
```

## System Architecture

### 1. Application Layer
```
app_CCB.py (Main Application)
├── Route Handlers
├── ML Integration
└── Database Operations
```

### 2. NLP Components
```
nlp_components/
├── book_chatbot.py (Chatbot Logic)
├── vector_store.py (Embedding Storage)
└── chat_routes.py (Chat Endpoints)
```

### 3. Database Layer
```
Database Operations
├── Schema Management
├── Vector Storage
└── Query Interface
```

## Key Features

### 1. Intelligent Book Recommendations
- Hybrid recommendation system
- Semantic similarity matching
- Natural language interaction
- Contextual explanations

### 2. Vector Search Capabilities
- FAISS-powered similarity search
- BERT embeddings for semantic understanding
- Optimized similarity scoring
- Efficient vector storage

### 3. Database Management
- Web-based database portal
- SQL query interface
- Schema migration tools
- BLOB data handling

### 4. Error Handling & Logging
- Comprehensive error tracking
- Graceful degradation
- Detailed logging system
- Debug information

## Deployment Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- Ollama service running locally
- SQLite3 database system

### Environment Setup
```bash
# Virtual Environment
python -m venv llm-book
source llm-book/bin/activate  # Unix
llm-book\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt

# Ollama Setup
ollama pull llama3.2:3b
```

### Configuration
```python
# Environment Variables (Optional)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

## Usage Instructions

### 1. Starting the Application
```bash
python app_CCB.py
```

### 2. Accessing Interfaces
- Main Interface: `http://localhost:5000`
- Chat Interface: `http://localhost:5000/chat`
- DB Portal: `http://localhost:5000/db_portal`

### 3. Using the Chatbot
- Natural language queries
- Book-specific questions
- Theme-based recommendations
- Style comparisons

## Best Practices

### 1. Development
- Follow PEP 8 style guide
- Document all functions
- Use type hints
- Implement error handling

### 2. Database
- Regular backups
- Safe migrations
- Query optimization
- Data validation

### 3. Testing
- Unit tests
- Integration tests
- Error scenario testing
- Performance monitoring

## Future Roadmap

### 1. Technical Improvements
- Vector compression
- Batch processing
- Caching layer
- Connection pooling

### 2. Feature Enhancements
- User feedback system
- Personalization
- Recommendation diversity
- Advanced analytics

### 3. Monitoring
- Performance metrics
- Health checks
- Usage analytics
- Error tracking

## Support and Maintenance

### 1. Logging
- Application logs
- Error tracking
- Performance monitoring
- User interaction logs

### 2. Troubleshooting
- Connection issues
- Database problems
- Model errors
- API failures

### 3. Updates
- Regular dependency updates
- Security patches
- Model improvements
- Feature enhancements 