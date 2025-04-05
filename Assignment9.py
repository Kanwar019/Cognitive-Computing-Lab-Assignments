import nltk
import re
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')

print("Question 1")

text = """Artificial Intelligence (AI) is revolutionizing the world. With self-driving cars, smart assistants, and AI-powered healthcare, the future looks exciting. AI's capabilities are expanding daily. However, ethical concerns and job displacement issues must be addressed. Overall, the AI-driven era promises efficiency and innovation."""

# Preprocessing
text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))

# Tokenization using Treebank tokenizer
tokenizer = TreebankWordTokenizer()
words = tokenizer.tokenize(text_clean)

# Manual sentence split (approximation)
sentences = text.split(". ")
print("Words:", words)
print("Sentences:", sentences)

# Stopwords removal
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w not in stop_words]
print("Filtered Words:", filtered_words)

# Word frequency
word_freq = Counter(filtered_words)
print("Word Frequency (Excluding Stopwords):")
print(word_freq)

print("\nQuestion 2")

# Stemming
porter = PorterStemmer()
lancaster = LancasterStemmer()
print("PorterStemmer:", [porter.stem(w) for w in filtered_words])
print("LancasterStemmer:", [lancaster.stem(w) for w in filtered_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("Lemmatized Words:", [lemmatizer.lemmatize(w) for w in filtered_words])

print("\nQuestion 3")

# a. Words with more than 5 letters
words_gt5 = re.findall(r'\b\w{6,}\b', text)
print("Words with more than 5 letters:", words_gt5)

# b. Extract numbers
numbers = re.findall(r'\b\d+\b', text)
print("Numbers:", numbers)

# c. Capitalized words
capitalized = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
print("Capitalized Words:", capitalized)

# d. Alpha-only words
alpha_only = re.findall(r'\b[a-zA-Z]+\b', text)
print("Alpha-only Words:", alpha_only)

# e. Words starting with vowels
vowels = [w for w in alpha_only if w[0].lower() in 'aeiou']
print("Words starting with vowels:", vowels)

print("\nQuestion 4")

# Custom tokenizer
def custom_tokenizer(text):
    text = re.sub(r'[^\w\s\'-\.]', '', text)
    tokens = re.findall(r"\d+\.\d+|\w+(?:-\w+)*|'\w+|\w+", text)
    return tokens

custom_tokens = custom_tokenizer(text)
print("Custom Tokens:", custom_tokens)

# Substitutions
subbed_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)
subbed_text = re.sub(r'https?://\S+|www\.\S+', '<URL>', subbed_text)
subbed_text = re.sub(r'\+?\d{1,3}[-\s]?\d{10}|\d{3}[-\s]\d{3}[-\s]\d{4}', '<PHONE>', subbed_text)

print("Text after substitutions:")
print(subbed_text)
