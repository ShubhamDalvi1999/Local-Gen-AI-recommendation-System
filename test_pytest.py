"""
The pytest module of Classy Classic Book allows a developer to test the module app_CCB, namely its functions which render the HTML pages. It contains the following functions:
    *test_index to test rendering of the respective page 
    *test_about_project to test rendering of the respective page 
    *test_contacts to test rendering of the respective page 
    *test_form to test rendering of the respective page 
"""
import pytest
from app_CCB import app

def test_index():
    """
    The function to test rendering of the respective page 
    """
    response = app.test_client().get('/')
    assert response.status_code == 200

def test_about_project():
    """
    The function to test rendering of the respective page 
    """
    response = app.test_client().get('/about_project')
    assert response.status_code == 200

def test_contacts():
    """
    The function to test rendering of the respective page 
    """
    response = app.test_client().get('/contacts')
    assert response.status_code == 200

def test_form():
    """
    The function to test rendering of the respective page 
    """
    response = app.test_client().get('/form')
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main()
    