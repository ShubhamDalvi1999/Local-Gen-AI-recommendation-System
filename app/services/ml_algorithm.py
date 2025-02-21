"""
Machine learning algorithm for book recommendations using SVM
"""

import numpy as np
from sklearn import svm
import pickle as pkl
import logging
import os
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

VECTOR_DIMENSION = 13
CONFIDENCE_THRESHOLD = 0.6

def get_dataset_categories():
    """Get unique categories from the dataset"""
    try:
        df = pd.read_csv(os.path.join('data', 'books', 'data.csv'))
        # Split categories and flatten the list
        all_categories = [cat.strip() for cats in df['categories'].dropna() for cat in cats.split(',')]
        # Get unique categories
        unique_categories = list(set(all_categories))
        return unique_categories
    except Exception as e:
        logger.error(f"Error getting dataset categories: {e}")
        # Return default categories if dataset is not accessible
        return ['Fiction', 'Mystery', 'Romance', 'Science Fiction', 'Fantasy', 'Historical Fiction']

def create_synthetic_data():
    """Create synthetic training data for the SVM classifier using actual dataset categories"""
    try:
        # Get categories from dataset
        categories = get_dataset_categories()
        
        # Create base vectors for each category
        base_vectors = {}
        for category in categories:
            # Initialize balanced base vector
            base = np.array([3] * VECTOR_DIMENSION)
            category_lower = category.lower()
            
            # 1. Reading pace (Vector[0])
            if any(x in category_lower for x in ['thriller', 'mystery', 'adventure']):
                base[0] = 1  # Fast-paced
            elif any(x in category_lower for x in ['literary', 'philosophy']):
                base[0] = 3  # Slow and contemplative
            else:
                base[0] = 2  # Moderate
            
            # 2. Character complexity (Vector[1])
            if any(x in category_lower for x in ['literary', 'drama', 'psychology']):
                base[1] = 1  # Complex characters
            elif 'adventure' in category_lower:
                base[1] = 3  # Straightforward
            else:
                base[1] = 2  # Balanced
            
            # 3. Descriptive passages (Vector[2])
            if any(x in category_lower for x in ['literary', 'historical', 'fantasy']):
                base[2] = 1  # Rich descriptions
            elif any(x in category_lower for x in ['thriller', 'mystery']):
                base[2] = 3  # Minimal
            else:
                base[2] = 2  # Moderate
            
            # 4. Themes (Vector[3])
            if 'romance' in category_lower:
                base[3] = 1  # Love and relationships
            elif any(x in category_lower for x in ['social', 'political']):
                base[3] = 2  # Social commentary
            elif any(x in category_lower for x in ['adventure', 'action']):
                base[3] = 3  # Adventure and discovery
            else:
                base[3] = 4  # Philosophy and morality
            
            # 5. Historical settings (Vector[4])
            if 'historical' in category_lower:
                base[4] = 1  # Love historical detail
            elif any(x in category_lower for x in ['contemporary', 'modern']):
                base[4] = 3  # Prefer timeless
            else:
                base[4] = 2  # Don't mind either
            
            # 6. Story length (Vector[5])
            if any(x in category_lower for x in ['epic', 'saga']):
                base[5] = 3  # Long
            elif 'novella' in category_lower:
                base[5] = 1  # Short
            else:
                base[5] = 2  # Medium
            
            # 7. Romantic subplots (Vector[6])
            if 'romance' in category_lower:
                base[6] = 1  # Essential
            elif any(x in category_lower for x in ['action', 'thriller', 'horror']):
                base[6] = 3  # Minimal
            else:
                base[6] = 2  # Nice but not necessary
            
            # 8. Writing style (Vector[7])
            if any(x in category_lower for x in ['poetry', 'literary']):
                base[7] = 1  # Flowery and poetic
            elif any(x in category_lower for x in ['comedy', 'humor']):
                base[7] = 3  # Witty and humorous
            else:
                base[7] = 2  # Clear and straightforward
            
            # 9. Tragic elements (Vector[8])
            if any(x in category_lower for x in ['tragedy', 'drama']):
                base[8] = 1  # Bring on tragedy
            elif 'comedy' in category_lower:
                base[8] = 3  # Happy endings
            else:
                base[8] = 2  # Some is fine
            
            # 10. Narrative perspective (Vector[9])
            if any(x in category_lower for x in ['memoir', 'autobiography', 'personal']):
                base[9] = 1  # First person
            elif any(x in category_lower for x in ['mystery', 'thriller']):
                base[9] = 2  # Third person limited
            else:
                base[9] = 3  # Third person omniscient
            
            # 11. Philosophical discussions (Vector[10])
            if any(x in category_lower for x in ['philosophy', 'psychology', 'metaphysical']):
                base[10] = 1  # Love philosophical content
            elif any(x in category_lower for x in ['action', 'thriller']):
                base[10] = 3  # Focus on story
            else:
                base[10] = 2  # Moderate amount
            
            # 12. Ending preference (Vector[11])
            if any(x in category_lower for x in ['mystery', 'thriller']):
                base[11] = 3  # Surprise endings
            elif 'literary' in category_lower:
                base[11] = 2  # Open to interpretation
            else:
                base[11] = 1  # Clear resolution
            
            # 13. Importance of humor (Vector[12])
            if any(x in category_lower for x in ['comedy', 'humor', 'satire']):
                base[12] = 1  # Very important
            elif any(x in category_lower for x in ['tragedy', 'drama']):
                base[12] = 3  # Not important
            else:
                base[12] = 2  # Nice to have
            
            base_vectors[category] = base
        
        # Generate variations with controlled random noise
        sample_x = []
        sample_y = []
        
        for category, base in base_vectors.items():
            # Generate variations per category
            for _ in range(100):
                # Add smaller noise to maintain more distinct categories
                noise = np.random.uniform(-0.2, 0.2, size=len(base))
                variation = base + noise
                variation = np.clip(variation, 1, 4)  # Ensure values stay within valid range
                sample_x.append(variation)
                sample_y.append(category)
        
        return np.array(sample_x), np.array(sample_y)
        
    except Exception as e:
        logger.error(f"Error creating synthetic data: {e}")
        raise

def train_classifier():
    """Train SVM classifier on synthetic data with improved parameters"""
    try:
        # Generate training data
        sample_x, sample_y = create_synthetic_data()
        
        # Initialize and train SVM with optimized parameters
        clf = svm.SVC(
            kernel='rbf',
            probability=True,
            C=2.0,
            gamma='auto',
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(sample_x, sample_y)
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save trained model
        model_path = os.path.join(data_dir, "classifier.pickle")
        with open(model_path, "wb") as f:
            pkl.dump(clf, f)
        
        logger.info(f"Successfully trained and saved classifier to {model_path}")
        return clf
        
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        raise

def validate_user_vector(vector):
    """Validate user input vector"""
    try:
        # Convert to numpy array if needed
        if not isinstance(vector, np.ndarray):
            vector = np.array(list(map(float, str(vector))))
        
        # Check dimension
        if vector.size != VECTOR_DIMENSION:
            raise ValueError(f"Input vector must have {VECTOR_DIMENSION} dimensions")
        
        # Check value range
        if not np.all((vector >= 1) & (vector <= 4)):
            raise ValueError("All values must be between 1 and 4")
        
        return vector
        
    except Exception as e:
        logger.error(f"Vector validation error: {e}")
        raise

def predict_book_category(user_vector):
    """Predict book category from user preferences with improved confidence handling"""
    try:
        # Validate input vector
        vector = validate_user_vector(user_vector)
        
        # Load or train classifier
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models')
        model_path = os.path.join(data_dir, "classifier.pickle")
        
        if not os.path.exists(model_path):
            logger.warning("Model not found, training new classifier")
            clf = train_classifier()
        else:
            with open(model_path, "rb") as f:
                clf = pkl.load(f)
        
        # Get prediction and probabilities
        vector = vector.reshape(1, -1)
        prediction = clf.predict(vector)
        probabilities = clf.predict_proba(vector)
        
        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_3_categories = [
            {
                'category': clf.classes_[idx],
                'confidence': float(probabilities[0][idx])
            }
            for idx in top_3_indices
        ]
        
        result = {
            'primary_category': top_3_categories[0]['category'],
            'confidence': top_3_categories[0]['confidence'],
            'is_confident': top_3_categories[0]['confidence'] >= CONFIDENCE_THRESHOLD,
            'alternative_categories': top_3_categories[1:]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting book category: {e}")
        raise

if __name__ == "__main__":
    # Train new classifier if run directly
    train_classifier() 