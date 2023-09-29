import unittest
from src.JaxDecompiler import decompiler


class MyTestCase(unittest.TestCase):
    def test_huge_number_of_vars(self):
        def python_f(s):
            limit = 1000
            l1 = [s for i in range(2)]
            for i in range(2, limit):
                l1.append(l1[i - 1] + l1[i - 2])
            return l1[-1]

        n = 10
        y_expected = python_f(n)
        decompiled_f = decompiler.python_jaxpr_python(python_f, (n,))
        y = decompiled_f(n)
        self.assertEqual(y, y_expected)


if __name__ == "__main__":
    unittest.main()
