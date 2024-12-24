# Recommendation System Implementation

## Overview
The recommendation system uses a Support Vector Machine (SVM) classifier implemented in two distinct phases: training and prediction. The system utilizes a pickle file to store and load the trained model efficiently.

## Phase 1: Model Training

### Training Process (ML_algorithm_CCB.py)
```python
def learn_clf():
    sample = create_syntetic()
    sample_x = sample[0]
    sample_y = sample[1]
    clf = svm.SVC()
    clf.fit(sample_x, sample_y)
    return clf
```

1. **Data Collection**
   - Downloads base vectors from SQLite database
   - Each vector represents a book's characteristics

2. **Synthetic Data Generation**
   - Creates variations of existing data
   - Adds controlled random noise
   - Expands training dataset

3. **Model Training**
   - Uses scikit-learn's SVM classifier
   - Trains on synthetic dataset
   - Optimizes for book classification

4. **Model Serialization**
   - Saves trained model to classifier.pickle
   - Uses Python's pickle module
   - Enables quick model loading

## Phase 2: Making Recommendations

### Prediction Process (app_CCB.py)
```python
def ML_execution(vector):
    with open("classifier.pickle", "rb") as f:
        clf = pkl.load(f)
    prediction = clf.predict([list(vector)])
    return prediction
```

1. **Model Loading**
   - Loads serialized model from classifier.pickle
   - Fast and efficient loading process
   - No need for retraining

2. **Prediction**
   - Processes user input vector
   - Makes prediction using loaded model
   - Returns book ID

3. **Book Retrieval**
   - Uses prediction to query database
   - Fetches book details and Amazon link
   - Returns recommendation to user

## Advantages of Pickle Implementation

1. **Performance Benefits**
   - Quick model loading
   - Efficient prediction process
   - No training overhead

2. **Separation of Concerns**
   - Training separate from prediction
   - Easy model updates
   - Modular architecture

3. **Maintenance**
   - Simple model versioning
   - Easy backup and restore
   - Efficient deployment 