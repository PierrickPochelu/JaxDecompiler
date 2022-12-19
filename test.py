import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    root_dir = './'
    suite = loader.discover(root_dir)
    runner = unittest.TextTestRunner()
    runner.run(suite)
