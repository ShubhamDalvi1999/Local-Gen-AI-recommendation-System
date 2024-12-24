"""
The app_CCB module of Classy Classic Book  renders the HTML pages (with CCS style), calls the ML algorithm function and downloads the data from the database. It contains the following functions:
    *data_extraction extracts the data (a book title and the Amazon link) from the database in accordance with the prediction (id), it requires the my_database database. 
    *ML_execution dowloads the clf class from the pickle file and uses it to predict the book. The function requires the classifier.pickle file. If updated, the ML_algorithm_CCB module changes the classifier.pickle file automatically. 
    *index to render of the respective page 
    *about_project to render the respective page 
    *contacts to render the respective page 
    *form to render the respective page 

The app_CCB module and all external modules are executed in the vertual environment. All required frameworks, libraries, modules, directories, templatetes, etc. are provided. 

To test the main module, two separate modules are available: test_unittest to test the functions which dowload and process the data and test_pytest to test rendering of the pages. If updated, the app_CCB module should be tested by both of the test modules. 

Updating the Machine Learning can be done in the external module ML. If updated, the ML_algorithm_CCB module changes the classifier.pickle file automatically. 
"""


from flask import Flask
from flask import render_template
from flask import request
import sqlite3
import pickle as pkl
from nlp_components.chat_routes import chat_bp
from db_portal import db_portal

app = Flask(__name__)

# Register blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(db_portal)

def data_extraction(prediction):
    """
    The function extracts the data (a book title and the Amazon link) from the database in accordance with the prediction (id), it requires the my_database database.
    Parameters
    ----------
    prediction : str
        The id of a predicted book
    Returns
    -------
    db_data : list
        The tuple of tuples containing a book title and the Amazon link
    """
    connection = sqlite3.connect('my_database.db')
    cursor = connection.cursor()
    cmd = f"SELECT book, link FROM Books WHERE id = {prediction}"
    cursor.execute(cmd)
    db_data = cursor.fetchall()
    connection.close()
    return db_data

def ML_execution(vector):
    """
    The function dowloads the clf class from the pickle file and uses it to predict the book. The function requires the classifier.pickle file. If updated, the ML_algorithm_CCB module changes the    classifier.pickle file automatically. 
    Parameters
    ----------
    vector : str
        The data to predict a book
    Returns
    -------
    prediction : str
        The id of a predicted book
    """
    with open ("classifier.pickle", "rb") as f:
        clf = pkl.load(f)
    prediction = clf.predict([list(vector)])
    return prediction


@app.route('/')
def index():
    """The function renders the index page""" 
    return render_template('index.html')

@app.route('/about_project')
def about_project():
    """The function renders the project page""" 
    return render_template('about_project.html')

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    """The function renders the form, accepts the answers from the web page, calls the prediction function, dowloads the dynamic data to render the recommendation from the database and renders the dynamic page with the recommendation
    """
    server_message = ''
    client_message = ''
    vector = ""
    if request.method == 'POST':
        for i in range(1,14):
            option = request.form.get(f"option{i}")
            vector = vector+str(option)
        if "on" in vector:
            #The error page is rendered because not all of the questions are answered
            return error_alert()
        prediction_np = ML_execution(vector)
        prediction = str(prediction_np[0])
        book_data = data_extraction(prediction)  
	#The recommendation page is rendered
        return result(book_data)
    #The form page is rendered
    return render_template('form.html', message=server_message)


@app.route('/error')
def error_alert():
    """The function renders the error page""" 
    return render_template('error_alert.html')


@app.route('/result/<book>') 
def result(book_data):
    """The function renders the recommendation page with dynamic data
    Parameters
    ----------
    book_data : list
        The tuple of tuples containing a book title and the Amazon link
    """ 
    for row in book_data:
        book_name = row[0]
        book_link = f"{row[1]}"
    return render_template('recommendation.html', book_link=book_link, book_name=book_name)

if __name__ == "__main__":
   app.debug = True  # Enable debug mode
   app.run(host='0.0.0.0', port=5000)


