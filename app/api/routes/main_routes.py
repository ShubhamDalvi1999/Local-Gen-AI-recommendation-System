"""
Main routes for the application
"""

from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle as pkl
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

def data_extraction(book_idx):
    """Extract book information from the dataset."""
    try:
        df = pd.read_csv(os.path.join('data', 'books', 'data.csv'))
        book = df.iloc[book_idx]
        
        book_info = {
            'title': book['title'] if pd.notna(book['title']) else 'Title not available',
            'author': book['authors'] if pd.notna(book['authors']) else 'Author unknown',
            'description': book['description'] if pd.notna(book['description']) else 'No description available',
            'categories': book['categories'].split(',') if pd.notna(book['categories']) else [],
            'rating': round(float(book['average_rating']), 2) if pd.notna(book['average_rating']) else None,
            'image_url': book['thumbnail'] if pd.notna(book['thumbnail']) else None,
            'preview_link': book['preview_link'] if pd.notna(book['preview_link']) else None
        }
        return book_info
    except Exception as e:
        logger.error(f"Error extracting book data: {str(e)}")
        raise

def ML_execution(vector):
    """Predict a book based on user preferences."""
    try:
        with open("classifier.pickle", "rb") as f:
            clf = pkl.load(f)
        vector_list = [int(x) for x in vector]
        prediction = clf.predict([vector_list])[0]
        return prediction
    except Exception as e:
        logger.error(f"Error in ML prediction: {str(e)}")
        raise

@main_bp.route('/')
def index():
    """Render the index page"""
    return render_template('index.html')

@main_bp.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@main_bp.route('/about_project')
def about_project():
    """Render the project page"""
    return render_template('about_project.html')

@main_bp.route('/contacts')
def contacts():
    """Render the contacts page"""
    return render_template('contacts.html')

@main_bp.route('/form', methods=['GET', 'POST'])
def form():
    """Handle the survey form and book recommendations"""
    if request.method == 'POST':
        try:
            # Collect form data
            vector = ""
            for i in range(1,14):
                option = request.form.get(f"option{i}")
                vector = vector + str(option)
            
            # Check if all questions are answered
            if "on" in vector:
                return redirect(url_for('main.error_alert'))
            
            # Get prediction
            try:
                prediction = ML_execution(vector)
            except Exception as e:
                logger.error(f"ML prediction failed: {str(e)}")
                return render_template('error.html', 
                    message="Sorry, we couldn't process your preferences. Please try again.")
            
            # Get book data
            try:
                book_info = data_extraction(prediction)
                if not book_info:
                    return render_template('error.html',
                        message="Sorry, we couldn't find a matching book. Please try again.")
                return render_template('recommendation.html', book_info=book_info)
            except Exception as e:
                logger.error(f"Database extraction failed: {str(e)}")
                return render_template('error.html',
                    message="Sorry, we couldn't retrieve the book information. Please try again.")
            
        except Exception as e:
            logger.error(f"Form processing error: {str(e)}")
            return render_template('error.html',
                message="An unexpected error occurred. Please try again.")
    
    return render_template('form.html')

@main_bp.route('/error')
def error_alert():
    """Render the error page"""
    return render_template('error_alert.html') 