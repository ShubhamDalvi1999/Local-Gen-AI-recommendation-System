"""
The ML_algorithm module is used to provide the clf object (saved in a classifier.pickle file) for app_CCB. The ML_algorithm module is not called each time when the app_CCB module runs but can be updated or executed with the updated my_database if needed. If updated, the ML_algorithm_CCB module changes the classifier.pickle file automatically. The ML_algorithm module requires my_database. The ML_algorithm module contains the following functions:
    *download_db to download the current data from the database
    *randomize to create noise for the downloaded data
    *create_syntetic to add the noise to the dowloaded data and creates syntectic data on the basis of the downloaded
    *learn_clf to execute the learning process for the algorithm using the synthetic data and return the clf object which will be used by the app_CCB module
    *create_pickle to create the pickle file with the clf object in classifier.pickle file so that it can be used by the app_CCB module at any time without the need to run the ML_algorithm module
"""

from random import randint
from sklearn import svm
import numpy as np
import pickle as pkl
import sqlite3


def download_db():
  """
  The function downloads the current data from the database
  """
  connection = sqlite3.connect('my_database.db')
  cursor = connection.cursor()
  cursor.execute('SELECT vector FROM Books')
  results = cursor.fetchall()
  db_dict = {}
  for i in range(14):
    db_dict[i+1] = list(str(results[i][0]))
  connection.close()
  return db_dict

def randomize(row):
    """
    The function creates noise for the downloaded data
    """
    sample = []
    for i in range(10):
        new_row = []
        for j in row:
            if j == "0":
                j = randint(1,4) 
            new_row.append(int(j))  
        sample.append(new_row)
    return sample

def create_syntetic():
    """
    The function adds the noise to the dowloaded data and creates syntectic data on the basis of the downloaded
    """
    sample_y = []
    sample_x = []
    db_dict = download_db()
    for i in db_dict:
        r = randomize(db_dict[i])
        sample_y.extend([i] * len(r))
        sample_x.extend(r)
    sample_y = np.array(sample_y)
    sample_x = np.array(sample_x)
    return (sample_x, sample_y)

def learn_clf():
    """
    The function executes the learning process using the synthetic data and returns the clf object which will be used by the app_CCB module
    """
    sample = create_syntetic()
    sample_x = sample[0]
    sample_y = sample[1]
    clf = svm.SVC()
    clf.fit(sample_x, sample_y)
    return clf

def create_pickle(clf):
    """
    The function creates the pickle file with the clf object in classifier.pickle file so that it can be used by the app_CCB module at any time without the need to run the ML_algorithm module
    """
    with open ("classifier.pickle", "wb") as f:
        pkl.dump(clf, f)

if __name__ == "__main__":
    clf = learn_clf()
    create_pickle(clf)






