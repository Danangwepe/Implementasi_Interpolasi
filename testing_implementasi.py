import numpy as np
import matplotlib.pyplot as plt
import unittest

# Data yang diberikan
x_points = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_points = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Fungsi Interpolasi Polinom Lagrange
def lagrange_interpolation(x, y, xi):
    def L(k, xi):
        terms = [(xi - x[j]) / (x[k] - x[j]) for j in range(len(x)) if j != k]
        return np.prod(terms, axis=0)
    
    yi = sum(y[k] * L(k, xi) for k in range(len(x)))
    return yi

# Fungsi Interpolasi Polinom Newton
def newton_interpolation(x, y, xi):
    def divided_diff(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y

        for j in range(1, n):
            for i in range(n-j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
        
        return coef[0, :]
    
    coef = divided_diff(x, y)
    n = len(coef)
    yi = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (xi - x[j])
        yi += term
    return yi

# Rentang nilai x untuk plot
x_range = np.linspace(5, 40, 400)

# Interpolasi menggunakan kedua metode
y_lagrange = [lagrange_interpolation(x_points, y_points, xi) for xi in x_range]
y_newton = [newton_interpolation(x_points, y_points, xi) for xi in x_range]

# Plot hasil interpolasi
plt.figure(figsize=(12, 6))
plt.plot(x_points, y_points, 'o', label='Data Points')
plt.plot(x_range, y_lagrange, label='Lagrange Interpolation')
plt.plot(x_range, y_newton, label='Newton Interpolation', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolasi Polinom Lagrange dan Newton')
plt.legend()
plt.grid(True)
plt.show()

# Unit test untuk memeriksa hasil interpolasi
class TestInterpolationMethods(unittest.TestCase):

    def test_lagrange_interpolation(self):
        # Test dengan nilai xi yang diketahui hasilnya
        self.assertAlmostEqual(lagrange_interpolation(x_points, y_points, 10), 30.0, places=5)
        self.assertAlmostEqual(lagrange_interpolation(x_points, y_points, 20), 40.0, places=5)
        self.assertAlmostEqual(lagrange_interpolation(x_points, y_points, 25), 18.0, places=5)

    def test_newton_interpolation(self):
        # Test dengan nilai xi yang diketahui hasilnya
        self.assertAlmostEqual(newton_interpolation(x_points, y_points, 10), 30.0, places=5)
        self.assertAlmostEqual(newton_interpolation(x_points, y_points, 20), 40.0, places=5)
        self.assertAlmostEqual(newton_interpolation(x_points, y_points, 25), 18.0, places=5)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
