import unittest
import os

if __name__ == "__main__":
    test_root_dir = os.path.dirname(os.path.realpath(__file__))
    print("Run all unit tests in the folder: ", test_root_dir)

    loader = unittest.TestLoader()
    suite = loader.discover(test_root_dir)
    runner = unittest.TextTestRunner()
    runner.run(suite)
