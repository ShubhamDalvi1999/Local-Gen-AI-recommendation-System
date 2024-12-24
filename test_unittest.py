"""
The unittest module of Classy Classic Book allows a developer to test the module app_CCB, namely, if its functions dowload and process the data correctly. It contains the following functions:
    *test_ML to test wheather the ML algorithm works correctly
    *test_extraction to test wheather the function downloads the data from the database correctly 
"""
import unittest
from app_CCB import ML_execution, data_extraction


class TestClassyBook(unittest.TestCase):
    def test_ML(self):
        """
        The function to test wheather the ML algorithm works correctly
        """
        self.assertEqual(ML_execution("1124243100339"), [13])
    def test_extraction(self):
        """
        The function to test wheather the functions downloads the data from the database correctly 
        """
        self.assertEqual((data_extraction("13"))[0][0], "Waiting for Godot")

if __name__ == "__main__":
    unittest.main()
    