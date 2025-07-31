from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing functions
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Intents and responses
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "exit"],
    "name": ["what is your name", "who are you"],
    "function": ["what can you do", "how can you help me"],
    "thanks": ["thank you", "thanks"]
}

responses = {
    "greeting": ["Hello! How can I help you?", "Hi there! What can I do for you?"],
    "goodbye": ["Goodbye! Have a nice day!", "See you soon!"],
    "name": ["I'm a chatbot created to assist you.", "You can call me your support bot."],
    "function": ["I can answer questions and assist with information.", "I help users navigate our services."],
    "thanks": ["You're welcome!", "Glad I could help!"]
}

faq_data = [
    "What is AI?",
    "AI stands for Artificial Intelligence.",
    "What is machine learning?",
    "Machine learning is a way computers learn from data.",
    "What is Python?",
    "Python is a programming language."
]
sent_tokens = [s.lower() for s in faq_data]

# Detect intent
def detect_intent(text):
    text = text.lower()
    for intent, keywords in intents.items():
        if any(keyword in text for keyword in keywords):
            return intent
    return None

# Fallback with TF-IDF
def generate_fallback(user_input):
    sent_tokens.append(user_input)
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    similarity_scores = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity_scores.argsort()[0][-2]
    flat_scores = similarity_scores.flatten()
    flat_scores.sort()
    best_score = flat_scores[-2]
    sent_tokens.pop()

    if best_score == 0:
        return "I'm not sure I understand. Can you rephrase?"
    else:
        return faq_data[idx]

# Chatbot response
def get_response(user_input):
    intent = detect_intent(user_input)
    if intent:
        return random.choice(responses[intent])
    else:
        return generate_fallback(user_input)

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]
    print("User:", user_message)  # ✅ For debugging
    bot_response = get_response(user_message)
    print("Bot:", bot_response)   # ✅ See if response is generated
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)
