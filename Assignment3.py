import pandas as pd
import random

# Question 1: Create the dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George', 'Hannah', 'Ian', 'Jane'],
    'Age': [25, 30, 35, 28, 40, 32, 38, 27, 45, 29],
    'Department': ['HR', 'IT', 'IT', 'Marketing', 'Sales', 'HR', 'IT', 'Marketing', 'Sales', 'HR'],
    'Salary': [50000, 60000, 65000, 55000, 70000, 52000, 62000, 53000, 72000, 51000]
}
df = pd.DataFrame(data)

# Question 2: Locate row 0, 4, 7, and 8
print(df.loc[[0, 4, 7, 8]])

# Question 3
print(df.iloc[3:8])  # (i) rows 3 to 7
print(df.iloc[4:9, 2:5])  # (ii) rows 4 to 8, columns 2 to 4
print(df.iloc[:, 1:4])  # (iii) all rows, columns 1 to 3

# Question 4: Simulated Iris dataset
iris_data = pd.DataFrame({
    'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
    'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
    'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
    'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
    'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
})
print(iris_data)

# Question 5: Delete row 4 and column 3
iris_modified = iris_data.drop(index=4).drop(iris_data.columns[3], axis=1)
print(iris_modified)

# Question 6: Employee dataset
employee_data = {
    'Employee_ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward'],
    'Department': ['HR', 'IT', 'IT', 'Marketing', 'Sales'],
    'Age': [29, 34, 41, 28, 38],
    'Salary': [50000, 70000, 65000, 55000, 60000],
    'Years_of_Experience': [4, 8, 10, 3, 12],
    'Joining_Date': ['2020-03-15', '2017-07-19', '2013-06-01', '2021-02-10', '2010-11-25'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Male'],
    'Bonus': [5000, 7000, 6000, 4500, 5000],
    'Rating': [4.5, 4.0, 3.8, 4.7, 3.5]
}
emp_df = pd.DataFrame(employee_data)

# (a) Shape
print("Shape:", emp_df.shape)

# (b) Info
print(emp_df.info())

# (c) Describe
print(emp_df.describe())

# (d) First 5 rows
print(emp_df.head())

# (e) Last 3 rows
print(emp_df.tail(3))

# (f) Average salary
print("Average Salary:", emp_df['Salary'].mean())

# (g) Total bonus
print("Total Bonus:", emp_df['Bonus'].sum())

# (h) Youngest age
print("Youngest Employee Age:", emp_df['Age'].min())

# (i) Highest rating
print("Highest Rating:", emp_df['Rating'].max())

# (j) Sorted by salary descending
print(emp_df.sort_values(by='Salary', ascending=False))

# (k) Add "Performance_Category" based on rating
def rate_category(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Good"
    else:
        return "Average"

emp_df['Performance_Category'] = emp_df['Rating'].apply(rate_category)
print(emp_df[['Name', 'Rating', 'Performance_Category']])

# (l) Check for missing values
print("Missing values:\n", emp_df.isnull())

# (m) Rename column "Employee_ID" to "ID"
emp_df.rename(columns={'Employee_ID': 'ID'}, inplace=True)
print(emp_df.columns)

# (n) Employees with more than 5 years experience and from IT
print(emp_df[(emp_df['Years_of_Experience'] > 5) & (emp_df['Department'] == 'IT')])

# (o) Add column "Tax" = 10% of salary
emp_df['Tax'] = emp_df['Salary'] * 0.10
print(emp_df[['Name', 'Salary', 'Tax']])
