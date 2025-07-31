import nltk
import numpy as np
import random
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Define intents and responses
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "exit"],
    "name": ["what is your name", "who are you"],
    "function": ["what can you do", "how can you help me", "your purpose"],
    "thanks": ["thank you", "thanks", "thanks a lot"]
}

responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?"],
    "goodbye": ["Goodbye! Have a great day!", "See you soon. Stay safe!"],
    "name": ["I am an AI assistant created to help you.", "You can call me ChatBot!"],
    "function": ["I can answer your questions and provide support.", "I'm here to help you with information and guidance."],
    "thanks": ["You're welcome!", "Glad I could help!"]
}

# Sample FAQ dataset for TF-IDF response fallback
faq_data = [
    "What is AI?",
    "AI stands for Artificial Intelligence.",
    "How does machine learning work?",
    "Machine learning uses data to train models that make predictions.",
    "What is Python?",
    "Python is a popular programming language."
]

sent_tokens = [sentence.lower() for sentence in faq_data]

# Intent classifier (rule-based)
def detect_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in user_input:
                return intent
    return None

# Fallback response generator using TF-IDF
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
        return "I'm not sure how to help with that. Could you ask something else?"
    else:
        return faq_data[idx]

# Main chatbot response function
def get_response(user_input):
    intent = detect_intent(user_input)
    if intent:
        return random.choice(responses[intent])
    else:
        return generate_fallback(user_input)

# Run chatbot in terminal
if __name__ == "__main__":
    print("Bot: Hello! I'm your smart assistant. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'bye']:
            print("Bot:", random.choice(responses['goodbye']))
            break
        else:
            print("Bot:", get_response(user_input))
