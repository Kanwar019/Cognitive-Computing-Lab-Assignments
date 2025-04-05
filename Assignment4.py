import numpy as np

# Q1: Create a 1D NumPy array with 5 elements and perform operations
arr1 = np.array([1, 2, 3, 4, 5])
print("Original:", arr1)
print("Addition of 2:", arr1 + 2)
print("Multiplication by 3:", arr1 * 3)
print("Division by 2:", arr1 / 2)

# Q2a: Reverse the NumPy array
arr2 = np.array([1, 2, 3, 6, 4, 5])
print("Reversed array:", arr2[::-1])

# Q2b: Most frequent value and its indices
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])

def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    most_common = unique[counts == max_count]
    indices = np.where(np.isin(arr, most_common))[0]
    return most_common, indices

x_common, x_indices = most_frequent(x)
print("Most frequent in x:", x_common, "Indices:", x_indices)

y_common, y_indices = most_frequent(y)
print("Most frequent in y:", y_common, "Indices:", y_indices)

# Q3: Accessing 2D array elements
arr3 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("1st row, 2nd col:", arr3[0, 1])
print("3rd row, 1st col:", arr3[2, 0])

# Q4: 1-D NumPy array with 25 numbers from 10 to 100 using linspace
my_array = np.linspace(10, 100, 25)
print("Array:", my_array)
print("Dimensions:", my_array.ndim)
print("Shape:", my_array.shape)
print("Total elements:", my_array.size)
print("Data type:", my_array.dtype)
print("Bytes consumed:", my_array.nbytes)

# Transpose using reshape
reshaped = my_array.reshape(25, 1)
print("Transpose using reshape():", reshaped)

# Transpose using T
transpose_T = my_array.T  # Has no effect on 1D arrays
print("Transpose using T:", transpose_T)

# Q5: 2D array of 3x4
ucs420_kanwarajaybir = np.array([[10, 20, 30, 40],
                                 [50, 60, 70, 80],
                                 [90, 15, 20, 35]])
print("Original 2D Array:\n", ucs420_kanwarajaybir)
print("Mean:", np.mean(ucs420_kanwarajaybir))
print("Median:", np.median(ucs420_kanwarajaybir))
print("Max:", np.max(ucs420_kanwarajaybir))
print("Min:", np.min(ucs420_kanwarajaybir))
print("Unique elements:", np.unique(ucs420_kanwarajaybir))

# Reshape to 4x3
reshaped_ucs420 = ucs420_kanwarajaybir.reshape(4, 3)
print("Reshaped to 4x3:\n", reshaped_ucs420)

# Resize to 2x3
resized_ucs420 = np.resize(ucs420_kanwarajaybir, (2, 3))
print("Resized to 2x3:\n", resized_ucs420)
