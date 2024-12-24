##Classy Classic Book(CCB)

##Table of contents

- Introduction
- Technologies
- Project setup


##Introduction

Many people want to get acquainted with classic literature but do not know how to choose books. If they pick books randomly or rely on influencers, the books they start to read can appear too remote from their own everyday problems and, as a result, not beneficial in terms of their self-development. Reading poorly chosen books can lead to waste of time and disappointment in classic literature.  

The application Classy Classic Books will help to choose the right classic books for everyone. To receive a list of classic books, a user fills in the form by answering ten questions and submits it for analysis. The Machine Learning algorithms of Classy Classic Books analyse the answers of the user, select a set of books suitable for him or her and display a final webpage with previews of these books from Amazon. The user can click on the displayed previews to buy the chosen books on Amazon.

##Technologies

#General description



Several programming languages are used while Python is chosen as the main language of the project. Python and its libraries and frameworks are chosen due to their ability to fulfil all important for the project functions, such as rendering the HTML templates, receiving responses from web applications, reading and writing data from a database, training ML algorithms.  

By the time the application starts to work, the SQLite database (my_database) is filled with the required data (id, title, Amazon link) and the algorithm (Support Vector Machines: C-Support Vector Classification in the ML_algorithm_CCB) is trained while an instance of sklearn.svm.SVC is serialized by the pickle module (as clf) for further use in the main Flask script (app_CCB). C-Support Vector Classification is trained using the synthetic data (140 samples, 13 features (questions), 14 possible classes) but in real-world case, the data is supposed to be collected by interviewing respondents.  

The Flask backend (the app_CCB module) renders all HTML templates and receives the data from the form when a user fills one. Each form question is translated into an integer while the whole form data received by the app_CCB module looks like that: 2322211121112. The Flask script downloads the “clf.predict” from "classifier.pickle" to predict the suitable book class, reads the book title and its Amazon link and use them to render a dynamic HTML page. The application and its virtual environment are located on Amazon Lightsail in a container.  

#The app_CCB module

The app_CCB module of Classy Classic Book  renders the HTML pages (with CCS style), calls the ML algorithm function and downloads the data from the database. It contains the following functions:
    data_extraction extracts the data (a book title and the Amazon link) from the database in accordance with the prediction (id), it requires the my_database database. 
    ML_execution dowloads the clf class from the pickle file and uses it to predict the book. The function requires the classifier.pickle file. If updated, the ML_algorithm_CCB module changes the classifier.pickle file automatically. 
    index to render of the respective page 
    about_project to render the respective page 
    contacts to render the respective page 
    form to render the respective page 

The app_CCB module and all external modules are executed in the virtual environment. All required frameworks, libraries, modules, directories, templates, etc. are provided. 

To test the main module, two separate modules are available: test_unittest to test the functions which download and process the data and test_pytest to test rendering of the pages. If updated, the app_CCB module should be tested by both of the test modules. 

Updating the Machine Learning can be done in the external module ML. If updated, the ML_algorithm_CCB module changes the classifier.pickle file automatically. 

#The ML_algorithm module

The ML_algorithm module is used to provide the clf object (saved in a classifier.pickle file) for app_CCB. The ML_algorithm module is not called each time when the app_CCB module runs but can be updated or executed with the updated my_database if needed. If updated, the ML_algorithm_CCB module changes the classifier.pickle file automatically. The ML_algorithm module requires my_database. The ML_algorithm module contains the following functions:
    *download_db to download the current data from the database
    *randomize to create noise for the downloaded data
    *create_syntetic to add the noise to the downloaded data and creates synthetic data on the basis of the downloaded
    *learn_clf to execute the learning process for the algorithm using the synthetic data and return the clf object which will be used by the app_CCB module
    *create_pickle to create the pickle file with the clf object in classifier.pickle file so that it can be used by the app_CCB module at any time without the need to run the ML_algorithm module


##Project setup

There is the compose.yaml, Dockerfile and the requirements file used for the Docker container.

Requirements: 

antiorm       1.2.1
blinker       1.8.2
click         8.1.7
colorama      0.4.6
db            0.1.1
db-sqlite3    0.0.1
Flask         3.0.3
iniconfig     2.0.0
itsdangerous  2.2.0
Jinja2        3.1.4
joblib        1.4.2
MarkupSafe    2.1.5
numpy         2.1.2
packaging     24.1
pip           23.2.1
pluggy        1.5.0
pytest        8.3.3
scikit-learn  1.5.2
scipy         1.14.1
setuptools    65.5.0
threadpoolctl 3.5.0
Werkzeug      3.0.3

How to run an application:
1. If downloaded from GitHub, the CCB application contains the  compose.yaml, Dockerfile and the requirements.txt files to build and run the CCB application. In the Docker terminal, one should go to the folder with CCB and execute the following command:  docker compose up
The CCB application will be run on a local host http://127.0.0.1:5000/ 
2. A user can install the required environment manually (the requirements are listed in the file requirements file), go to the folder with CCB in Command Prompt and (if the required tools mentioned above are installed and all the modules and parts of the application correctly downloaded) execute the following commands: set FLASK_APP=app_CCB.py and flask run. The application will be on the local host http://127.0.0.1:5000/

The CCB application can be reached via amazonlightsail.com (https://ccb-service.st2ff8qfgnvqm.eu-west-1.cs.amazonlightsail.com/), if CCB is deployed.








