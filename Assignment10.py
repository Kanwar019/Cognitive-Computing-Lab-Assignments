import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import markovify

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Q1 - Tokenization and Word Frequency
text = "Artificial Intelligence is transforming industries with powerful tools. It helps automate tasks, optimize operations, and improve decision-making. From healthcare to finance, AI is making an impact. Machine Learning, a subfield of AI, enables systems to learn from data. Natural Language Processing is a crucial part of AI for human-computer interaction. The future with AI looks promising."

text_clean = re.sub(r'[^\w\s]', '', text.lower())

words_split = text_clean.split()
words_token = word_tokenize(text_clean)
sentences_token = sent_tokenize(text)

print("Q1 - Python split():", words_split)
print()

print("Q1 - NLTK word_tokenize():", words_token)
print()

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_token if word not in stop_words]

word_freq = Counter(filtered_words)
print("Q1 - Word Frequency:", word_freq)
print()

# Q2 - Stemming and Lemmatization
alpha_only = re.findall(r'\b[a-zA-Z]+\b', text_clean)
filtered_alpha = [w for w in alpha_only if w not in stop_words]

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_alpha]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_alpha]

print("Q2 - Stemmed Words:", stemmed_words)
print()

print("Q2 - Lemmatized Words:", lemmatized_words)
print()

print("Q2 - Stemming is quicker and simpler but less accurate; lemmatization is more accurate and context-aware.")
print()

# Q3 - Bag of Words and TF-IDF
texts = [
    "iPhone 14 is amazing. Camera quality is mind-blowing!",
    "Samsung Galaxy delivers excellent performance and display.",
    "The Google Pixel is great for photos and software experience."
]

cv = CountVectorizer()
bow = cv.fit_transform(texts).toarray()
print("Q3 - Bag of Words:", bow)
print()

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(texts)

features = tfidf.get_feature_names_out()
for i, row in enumerate(tfidf_matrix.toarray()):
    top = row.argsort()[-3:][::-1]
    print(f"Q3 - Top 3 TF-IDF Keywords for Text {i+1}:")
    for idx in top:
        print(f"{features[idx]}: {row[idx]:.3f}")
    print()

# Q4 - Similarity Measures
text1 = "Artificial Intelligence is revolutionizing industries with automation."
text2 = "Blockchain ensures secure decentralized data management."

def preprocess(text):
    return [w for w in word_tokenize(re.sub(r'[^\w\s]', '', text.lower())) if w not in stop_words]

set1 = set(preprocess(text1))
set2 = set(preprocess(text2))

intersection = set1 & set2
union = set1 | set2
jaccard = len(intersection) / len(union)

vec = TfidfVectorizer()
tfidf_vecs = vec.fit_transform([text1, text2])
cos_sim = cosine_similarity(tfidf_vecs[0], tfidf_vecs[1])[0][0]

print(f"Q4 - Jaccard Similarity: {jaccard:.3f}")
print()

print(f"Q4 - Cosine Similarity: {cos_sim:.3f}")
print()

print("Q4 - Cosine similarity captures semantic distance better than Jaccard in this context.")
print()

# Q5 - Sentiment Analysis and Word Cloud
review = "This phone is amazing. The battery life is great and the camera is stunning."

blob = TextBlob(review)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

if polarity > 0:
    sentiment = "Positive"
elif polarity < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print(f"Q5 - Polarity: {polarity:.2f}")
print()

print(f"Q5 - Subjectivity: {subjectivity:.2f}")
print()

print(f"Q5 - Sentiment: {sentiment}")
print()

if sentiment == "Positive":
    wc = WordCloud(background_color='white').generate(review)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("Q5 - Word Cloud for Positive Review")
    plt.show()

# Q6 - Text Generation with Markov Chains
training_text = """Machine learning allows computers to learn from data.
It is a part of AI. This enables predictive models and intelligent decisions.
AI is transforming the way industries operate. Algorithms adapt to new data."""

text_model = markovify.Text(training_text)

print("Q6 - Generated Text:")
print()

for _ in range(3):
    sentence = text_model.make_sentence()
    print(sentence if sentence else "No sentence generated.")
print()
