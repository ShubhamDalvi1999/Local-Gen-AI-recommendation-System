Classy Classic Book(CCB)

The system combines both approaches:
Questionnaire → SVM Classifier → Direct book recommendation
Chat Interface → Vector Similarity → Multiple similar book recommendations
# Classy Books Web: AI-Powered Book Recommendation System

A modern book recommendation system that combines traditional Machine Learning with Large Language Models to provide personalized classic literature recommendations.

## 🚀 Features

- **Dual Recommendation Approaches**:
  - Questionnaire-based recommendations using SVM
  - Natural language chat interface using BERT embeddings
- **Interactive Web Interface**
- **Vector-based Book Matching**
- **SQLite Database with Vector Store**

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML/AI**: 
  - SVM for questionnaire processing
  - BERT embeddings for similarity matching
  - Ollama (llama3.2:3b) for chat interface
- **Database**: SQLite with FAISS vector store
- **Frontend**: HTML, CSS, JavaScript

## 📋 Prerequisites

- Python 3.8+
- Ollama installed and running
- SQLite3

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/classy_books_web.git
cd classy_books_web
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Ollama service**
```bash
ollama run llama3.2:3b
```

5. **Run the application**
```bash
python app_CCB.py
```

## 💡 Usage

1. **Questionnaire-Based Recommendations**:
   - Navigate to `/form`
   - Answer 13 questions about your preferences
   - Get personalized book recommendations

2. **Chat Interface**:
   - Click the chat icon or go to `/chat`
   - Describe your interests or ask for recommendations
   - Get AI-powered suggestions with similarity scores

3. **Database Portal**:
   - Access `/db_portal` for database exploration
   - Execute custom queries
   - View book information and vectors

## 🏗️ Project Structure

```
classy_books_web/
├── app_CCB.py              # Main Flask application
├── ML_algorithm_CCB.py     # SVM implementation
├── nlp_components/         # NLP and chat functionality
├── static/                 # CSS, JS, and assets
├── templates/             # HTML templates
└── my_database.db         # SQLite database
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

Shubham Dalvi - [@ShubhamDalvi1999](https://github.com/ShubhamDalvi1999)

## 🙏 Acknowledgments

- BERT model for embeddings
- Ollama for LLM capabilities
- FAISS for vector similarity search
- Flask for web framework








