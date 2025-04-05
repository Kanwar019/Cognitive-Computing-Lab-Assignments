import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Question 1: Sales Data Analysis")

# Part I: Data Generation
np.random.seed(102317223)
sales_array = np.random.randint(1000, 5001, size=(12, 4))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
columns = ['Electronics', 'Clothing', 'Home & Kitchen', 'Sports']
df = pd.DataFrame(sales_array, columns=columns, index=months)

# Part II: Data Manipulation
print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

df['Total Sales'] = df.sum(axis=1)
df['Monthly Total'] = df['Total Sales']
df['Growth Rate'] = df['Total Sales'].pct_change() * 100

# Average growth between months for each category
avg_growth = df[columns].pct_change().mean() * 100
print("\nAverage monthly growth (%) for each category:")
print(avg_growth)

# Apply discount
if 102317223 % 2 == 0:
    df['Electronics'] = df['Electronics'] * 0.9
else:
    df['Clothing'] = df['Clothing'] * 0.85

print("\nData after applying discount:")
print(df.head())

# Part III: Visualization
print("\nGenerating plots...")

plt.figure(figsize=(10, 6))
for col in columns:
    plt.plot(df.index, df[col], label=col)
plt.title("Monthly Sales Trend per Category")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df[columns])
plt.title("Box Plot of Monthly Sales per Category")
plt.grid(True)
plt.show()

print("\nQuestion 2")

array = np.array([[1, -2, 3], [-4, 5, -6]])
print("Original array:\n", array)

abs_val = np.abs(array)
print("Absolute values:\n", abs_val)

flat = array.flatten()
print("Flattened Percentiles: 25th =", np.percentile(flat, 25), ", 50th =", np.percentile(flat, 50), ", 75th =", np.percentile(flat, 75))

print("Column-wise percentiles:")
print("25th:\n", np.percentile(array, 25, axis=0))
print("50th:\n", np.percentile(array, 50, axis=0))
print("75th:\n", np.percentile(array, 75, axis=0))

print("Row-wise percentiles:")
print("25th:\n", np.percentile(array, 25, axis=1))
print("50th:\n", np.percentile(array, 50, axis=1))
print("75th:\n", np.percentile(array, 75, axis=1))

print("\nFlattened - Mean:", np.mean(flat), "Median:", np.median(flat), "Std Dev:", np.std(flat))
print("Column-wise - Mean:", np.mean(array, axis=0), "Median:", np.median(array, axis=0), "Std Dev:", np.std(array, axis=0))
print("Row-wise - Mean:", np.mean(array, axis=1), "Median:", np.median(array, axis=1), "Std Dev:", np.std(array, axis=1))

print("\nQuestion 3")

a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
print("Original:", a)
print("Floor:", np.floor(a))
print("Ceil:", np.ceil(a))
print("Truncated:", np.trunc(a))
print("Rounded:", np.round(a))

print("\nQuestion 4")

lst = [10, 20, 30, 40]
i, j = 1, 3
temp = lst[i]
lst[i] = lst[j]
lst[j] = temp
print(f"List after swapping index {i} and {j}:", lst)

print("\nQuestion 5")

s = {1, 2, 3, 4}
lst = list(s)
lst[0], lst[2] = lst[2], lst[0]
s = set(lst)
print("Set after swapping elements:", s)
