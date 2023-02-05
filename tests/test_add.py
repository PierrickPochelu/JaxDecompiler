import unittest
from src.JaxDecompiler import decompiler


class MyTestCase(unittest.TestCase):
    def test_from_strings_to_callable(self):
        lines = ["def f(x):", "    return x+1"]
        python_f = decompiler._from_strings_to_callable(lines)
        y = python_f(10)
        self.assertEqual(11, y)

    def test_add_2literals(self):
        def python_f():
            return 1 + 1

        decompiled_f = decompiler.python_jaxpr_python(python_f, ())
        y = decompiled_f()
        self.assertEqual(2, y)

    def test_multi_output(self):
        def python_f():
            return 1, 1

        decompiled_f = decompiler.python_jaxpr_python(python_f, ())
        y = decompiled_f()
        self.assertEqual((1, 1), y)

    def test_add_op_var_literal(self):
        def python_f(x):
            return x + 1

        decompiled_f = decompiler.python_jaxpr_python(python_f, (1,))
        y = decompiled_f(10)
        self.assertEqual(11, y)

    def test_add_op_2vars(self):
        def python_f(x, y):
            return x + y

        decompiled_f = decompiler.python_jaxpr_python(python_f, (1, 1))
        y = decompiled_f(10, 1)
        self.assertEqual(11, y)

    def test_add_op_3vars(self):
        def python_f(x, y, z):
            return x + y + z

        decompiled_f = decompiler.python_jaxpr_python(python_f, (1, 1, 1))
        y = decompiled_f(10, 100, 1)
        self.assertEqual(111, y)

    def test_add_op_3vars_3literals_4outputs(self):
        def python_f(x, y, z):
            return x + 1, y + z, 1 + 1, 3

        decompiled_f = decompiler.python_jaxpr_python(python_f, (1, 1, 1))
        y = decompiled_f(10, 100, 1)

        self.assertEqual(4, len(y))
        self.assertEqual((11, 101, 2, 3), y)

    def test_add_multiple_lines(self):
        def python_f(x, y, z):
            # if x==1,y==1,z==1
            x += 1  # x==2
            y += 10  # y==11
            z += x + y + 1  # z==1+2+11+1==15
            return x, y, z

        decompiled_f = decompiler.python_jaxpr_python(python_f, (1, 1, 1))
        y = decompiled_f(1, 1, 1)

        self.assertEqual(3, len(y))
        self.assertEqual((2, 11, 15), y)


if __name__ == "__main__":
    unittest.main()
