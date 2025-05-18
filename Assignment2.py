# Assignment 2 - Cognitive Computing (UCS420)
# List, Tuple, Set, Dictionary Operations

# 1. List Operations
L = [10, 20, 30, 40, 50, 60, 70, 80]
L.append(200)
L.append(300)
L.remove(10)
L.remove(30)
L.sort()
print("1. Ascending List:", L)
L.sort(reverse=True)
print("1. Descending List:", L)

# 2. Tuple Operations
scores = (45, 89.5, 76, 45.4, 89, 92, 58, 45)
max_score = max(scores)
print("2. Max Score:", max_score, "at index", scores.index(max_score))
min_score = min(scores)
print("2. Min Score:", min_score, "appears", scores.count(min_score), "times")
reversed_list = list(reversed(scores))
print("2. Reversed List:", reversed_list)
check = 76
if check in scores:
    print(f"2. {check} found at index", scores.index(check))
else:
    print(f"2. {check} is not present in the tuple")

# 3. Random Numbers and Count
import random
numbers = [random.randint(100, 900) for _ in range(100)]
odd_numbers = [x for x in numbers if x % 2 != 0]
even_numbers = [x for x in numbers if x % 2 == 0]
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True
prime_numbers = [x for x in numbers if is_prime(x)]
print("3. Odd Numbers:", odd_numbers)
print("3. Even Numbers:", even_numbers)
print("3. Prime Numbers:", prime_numbers)

# 4. Set Operations
A = {34, 56, 78, 90}
B = {78, 45, 90, 23}
print("4. Union:", A | B)
print("4. Intersection:", A & B)
print("4. Symmetric Difference:", A ^ B)
print("4. A is subset of B:", A.issubset(B))
print("4. B is superset of A:", B.issuperset(A))
X = int(input("4. Enter a score to remove from A: "))
if X in A:
    A.remove(X)
    print(f"4. {X} removed. A is now:", A)
else:
    print(f"4. {X} not found in A.")

# 5. Dictionary Key Rename
sampledict = {
    'name': 'Kelly',
    'age': '25',
    'salary': 8000,
    'city': 'New york'
}
sampledict['location'] = sampledict.pop('city')
print("5. Dictionary after renaming key:", sampledict)
