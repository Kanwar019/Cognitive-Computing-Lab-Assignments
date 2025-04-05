import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("Question 1")
M = float(input("Enter value for M: "))
x = np.linspace(-10, 10, 200)
y1 = M * x**2
y2 = M * np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y1, label="y = M·x²", color="blue", linestyle='-')
plt.plot(x, y2, label="y = M·sin(x)", color="red", linestyle='--')
plt.title("Mathematical Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

print("\nQuestion 2")
data = {'Subject': ['Math', 'Physics', 'Chemistry', 'Biology', 'English'],
        'Score': [85, 90, 78, 88, 92]}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
colors = sns.color_palette("Set2")
barplot = sns.barplot(x='Subject', y='Score', data=df, palette=colors)
for i, val in enumerate(df['Score']):
    barplot.text(i, val + 1, str(val), ha='center')
plt.title("Scores by Subject")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.grid(True, axis='y')
plt.show()

print("\nQuestion 3")
np.random.seed(102317223)
data = np.random.randn(50)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(np.cumsum(data))
axs[0, 0].set_title("Cumulative Sum")
axs[0, 0].set_xlabel("Index")
axs[0, 0].set_ylabel("Value")
axs[0, 0].grid(True)

axs[0, 1].scatter(range(50), data + np.random.randn(50) * 0.2, color='orange')
axs[0, 1].set_title("Scatter Plot with Noise")
axs[0, 1].set_xlabel("Index")
axs[0, 1].set_ylabel("Value")
axs[0, 1].grid(True)

fig.delaxes(axs[1, 0])
fig.delaxes(axs[1, 1])
plt.tight_layout()
plt.show()

print("\nQuestion 4")
df = pd.read_csv("company_sales_data.csv")

# 1. Total profit line plot
plt.figure(figsize=(8, 6))
plt.plot(df['month_number'], df['total_profit'], marker='o')
plt.title("Total Profit Over Months")
plt.xlabel("Month Number")
plt.ylabel("Total Profit")
plt.grid(True)
plt.show()

# 2. Multiline plot for product sales
plt.figure(figsize=(10, 6))
for column in df.columns[1:-1]:  # exclude month_number and total_profit
    plt.plot(df['month_number'], df[column], label=column)
plt.title("Sales Data of Products")
plt.xlabel("Month Number")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()

# 3. Bar chart for all features
df.drop('month_number', axis=1).sum().plot(kind='bar', figsize=(10, 6), colormap='tab20')
plt.title("Total Sales and Profit by Product")
plt.ylabel("Total Units / Profit")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.show()