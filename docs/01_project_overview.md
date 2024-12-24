# Project Overview and Architecture

## Overview
Classy Classic Books is a Flask-based web application that provides personalized book recommendations using machine learning. The system analyzes user preferences through a questionnaire and suggests classic books that match their interests.

## Architecture

### Core Components
```
classy_books_web/
├── app_CCB.py           # Main Flask application
├── ML_algorithm_CCB.py  # Machine learning logic
├── classifier.pickle    # Serialized ML model
├── my_database.db      # SQLite database
├── templates/          # HTML templates
└── static/            # Static assets (CSS, images)
```

### Supporting Components
- `test_unittest.py` & `test_pytest.py`: Test suites for functionality verification
- `Dockerfile` & `compose.yaml`: Container configuration for deployment
- `requirements.txt`: Python package dependencies

### Architecture Pattern
The application follows the MVC (Model-View-Controller) pattern:

1. **Model Layer**
   - Machine learning algorithm (ML_algorithm_CCB.py)
   - Database interactions (SQLite)
   - Serialized classifier model

2. **View Layer**
   - HTML templates with dynamic content
   - CSS styling for user interface
   - Responsive web design

3. **Controller Layer**
   - Flask routes in app_CCB.py
   - Form processing
   - Business logic coordination

### Key Features
- User preference collection through interactive form
- Machine learning-based recommendation system
- Direct links to purchase recommended books
- Containerized deployment support
- Comprehensive test coverage 