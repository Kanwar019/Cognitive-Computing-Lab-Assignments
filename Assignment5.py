import numpy as np
import matplotlib.pyplot as plt

print("Question 1")
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print("Sum of all elements:", gfg.sum())
print("Row-wise sum:\n", gfg.sum(axis=1))
print("Column-wise sum:\n", gfg.sum(axis=0))

print("\nQuestion 2(a)")
array_a = np.array([10, 52, 62, 16, 16, 54, 453])
print("Sorted array:", np.sort(array_a))
print("Indices of sorted array:", np.argsort(array_a))
print("4 smallest elements:", np.sort(array_a)[:4])
print("5 largest elements:", np.sort(array_a)[-5:][::-1])

print("\nQuestion 2(b)")
array_b = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
ints = array_b[np.equal(np.mod(array_b, 1), 0)]
floats = array_b[np.not_equal(np.mod(array_b, 1), 0)]
print("Integer elements only:", ints)
print("Float elements only:", floats)

print("\nQuestion 3")
X = ord('K') + ord('S')
sales = np.array([X, X+50, X+100, X+150, X+200])
print("Initial Sales:", sales)

tax_rate = ((X % 5) + 5) / 100
print("Tax Rate:", tax_rate)
taxed_sales = sales + (sales * tax_rate)
print("Taxed Sales:", taxed_sales)

discounted_sales = np.where(sales < X+100, sales * 0.95, sales * 0.90)
print("Discounted Sales:", discounted_sales)

weekly_sales = np.vstack([sales, sales * 1.02, sales * 1.04])
print("3 Weeks Sales with 2% increase per week:\n", weekly_sales)

print("\nQuestion 4")
x = np.linspace(-10, 10, 100)
y1 = x ** 2
y2 = np.sin(x)
y3 = np.exp(x)
y4 = np.log(np.abs(x) + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y1)
plt.title("Y = x²")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.title("Y = sin(x)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, y3)
plt.title("Y = eˣ")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, y4)
plt.title("Y = log(|x| + 1)")
plt.grid(True)

plt.tight_layout()
plt.show()