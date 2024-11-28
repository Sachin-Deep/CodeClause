import nltk
import re
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.metrics.distance import edit_distance  # for Levenshtein distance
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def word_tokenizer(data):
    return word_tokenize(data.lower())

def remove_noise(word_tokens):
    return [token for token in word_tokens if token not in stop_words and token not in punctuation]

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

# Define the patterns and responses
patterns = [
    (r'hi|hello|hey', ['Hi there!', 'Hello!', 'Hey!']),
    (r'bye|goodbye', ['Bye', 'Goodbye!', 'See you later!']),
    (r'how are you', ["I'm doing great, thank you!", "I'm here to chat with you!"]),
    (r'what is your name', ["I'm a chatbot!", "You can call me Chatbot."]),
    (r'help|assist', ["I'm here to help you!", "What do you need assistance with?"]),
]

# Expand patterns to include synonyms using WordNet
def expand_patterns_with_synonyms(patterns):
    expanded_patterns = []
    for pattern, responses in patterns:
        words = pattern.split('|')
        expanded_words = []
        for word in words:
            synonyms = get_synonyms(word)
            synonyms.add(word)
            expanded_words.append('|'.join(synonyms))
        expanded_pattern = '|'.join(expanded_words)
        expanded_patterns.append((expanded_pattern, responses))
    return expanded_patterns

expanded_patterns = expand_patterns_with_synonyms(patterns)

# Function to compute the best match using Levenshtein distance
def best_match_by_edit_distance(user_input, patterns):
    min_distance = float('inf')
    best_response = "I'm not sure I understand. Could you tell me more?"

    for pattern, responses in patterns:
       
        pattern_tokens = remove_noise(word_tokenizer(pattern.replace('|', ' ')))
        pattern_text = ' '.join(pattern_tokens)

        
        distance = edit_distance(user_input, pattern_text)

        
        if distance < min_distance:
            min_distance = distance
            best_response = random.choice(responses)

    return best_response

# Function to generate a response based on the closest matching pattern
def generate_response(user_input):
    
    user_input_tokens = remove_noise(word_tokenizer(user_input))
    cleaned_input = ' '.join(user_input_tokens)

   
    for pattern, responses in patterns:
        if re.fullmatch(pattern, cleaned_input):
            return random.choice(responses)
    
    
    return best_match_by_edit_distance(cleaned_input, expanded_patterns)


def chatbot():
    conversation_history = []
    while True:
        user_input = input("You: ")
        
        if re.search(r'bye|goodbye', user_input.lower()):
            print('Chatbot: Goodbye!')
            break
        
        chatbot_response = generate_response(user_input)
        conversation_history.append(user_input)
        
        print('Chatbot:', chatbot_response)


chatbot()
