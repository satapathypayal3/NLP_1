import os 
import re #regex
import nltk #nlp toolkit

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# STEP 1: LOAD DOCUMENTS

# Get the folder where this Python file is located
project_folder = os.path.dirname(os.path.abspath(__file__))

# Locate the dataset folder
dataset_folder = os.path.join(project_folder, "dataset")

documents = []

for filename in os.listdir(dataset_folder):

    if filename.endswith(".txt"):

        file_path = os.path.join(dataset_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(text)


print("\nNLP PIPELINE - DOCUMENT COLLECTION")

print("\nNumber of documents:", len(documents))


# STEP 2: COMBINE DOCUMENTS

corpus = "\n".join(documents)

print("Total characters:", len(corpus))


# STEP 3: SENTENCE TOKENIZATION

sentences = sent_tokenize(corpus)

print("Total sentences:", len(sentences))

print("\nSAMPLE SENTENCES:")
for sentence in sentences[:3]:
    print("-", sentence)


# STEP 4: WORD TOKENIZATION
#tokenization is the foundational process of splitting raw, unstructured text into smaller, manageable pieces called tokens

words = word_tokenize(corpus)

print("\nTotal tokens before preprocessing:", len(words))

print("\nSAMPLE TOKENS:")
print(words[:20])


# STEP 5: LOWERCASE

lowercase_words = [word.lower() for word in words]


# STEP 6: REMOVE PUNCTUATION AND SPECIAL CHARACTERS

clean_words = []

for word in lowercase_words:

    if re.fullmatch(r"[a-z]+", word):
        clean_words.append(word)


print("\nTokens after punctuation removal:", len(clean_words))


# STEP 7: REMOVE STOP WORDS

stop_words = set(stopwords.words("english"))

filtered_words = [
    word for word in clean_words
    if word not in stop_words
]

print("Tokens after stop-word removal:", len(filtered_words))


# STEP 8: LEMMATIZATION
# a text cleaning step in nlp that changes words into their true root or dictionary form, called lemma.

lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_words
]


# STEP 9: DISPLAY PREPROCESSED TEXT

print("\nPREPROCESSED TEXT")

print(lemmatized_words[:50])


# STEP 10: VOCABULARY

vocabulary = set(lemmatized_words)

print("\nVocabulary size:", len(vocabulary))


# STEP 11: BASIC CORPUS STATISTICS

print("\nCORPUS STATISTICS")

print("Documents:", len(documents))
print("Sentences:", len(sentences))
print("Original tokens:", len(words))
print("Clean tokens:", len(clean_words))
print("Tokens after stop-word removal:", len(filtered_words))
print("Vocabulary size:", len(vocabulary))


# STEP 12: SPELLING ERROR DETECTION AND CORRECTION

def levenshtein_distance(word1, word2): #ld measures how different two text words are by counting the least number of single-letter changes- such as additions, removals, or swaps- needed to turn one word into the other.
    """
    Calculate the Levenshtein edit distance between two words.
    """

    rows = len(word1) + 1
    cols = len(word2) + 1

    # Create distance matrix
    distance = [[0] * cols for _ in range(rows)]

    # Initialize first column
    for i in range(rows):
        distance[i][0] = i

    # Initialize first row
    for j in range(cols):
        distance[0][j] = j

    # Calculate edit distance
    for i in range(1, rows):

        for j in range(1, cols):

            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance[i][j] = min(
                distance[i - 1][j] + 1,       # deletion
                distance[i][j - 1] + 1,       # insertion
                distance[i - 1][j - 1] + cost # substitution
            )

    return distance[-1][-1]


def correct_spelling(word, vocabulary):
    """
    Find the closest word in the vocabulary.
    """

    word = word.lower()

    # If word already exists, no correction is required
    if word in vocabulary:
        return word, 0

    best_word = None
    best_distance = float("inf")

    for candidate in vocabulary:

        distance = levenshtein_distance(word, candidate)

        if distance < best_distance:
            best_distance = distance
            best_word = candidate

    return best_word, best_distance


# STEP 13: TEST SPELLING CORRECTION

print("\nSPELLING ERROR DETECTION AND CORRECTION")

# Deliberately misspelled words
misspelled_words = [
    "naturall",
    "langauge",
    "machne",
    "computr",
    "proccess"
]

for word in misspelled_words:

    corrected_word, distance = correct_spelling(
        word,
        vocabulary
    )

    print(
        f"{word} → {corrected_word} "
        f"(edit distance: {distance})"
    )

# STEP 14: AMBIGUITY HANDLING

print("\nAMBIGUITY HANDLING")


# Lexical Ambiguity
# The word "bank" can have different meanings.
#
# Meaning 1: financial institution
# Meaning 2: side of a river
#
# We use context words to determine the likely meaning.


bank_meanings = {
    "financial": {
        "bank",
        "money",
        "deposit",
        "withdraw",
        "account",
        "loan",
        "cash",
        "finance",
        "payment",
        "credit"
    },

    "river": {
        "bank",
        "river",
        "water",
        "shore",
        "fishing",
        "boat",
        "stream",
        "lake",
        "nature"
    }
}


def resolve_bank_ambiguity(sentence):
    """
    Resolve the meaning of the word 'bank'
    using contextual keywords.
    """

    words = set(
        re.findall(r"[a-z]+", sentence.lower())
    )

    financial_score = len(
        words.intersection(bank_meanings["financial"])
    )

    river_score = len(
        words.intersection(bank_meanings["river"])
    )

    if financial_score > river_score:
        meaning = "Financial institution"

    elif river_score > financial_score:
        meaning = "River bank"

    else:
        meaning = "Ambiguous / insufficient context"

    return meaning, financial_score, river_score


# Test examples

ambiguous_sentences = [
    "I went to the bank to deposit money.",
    "The children played near the bank of the river.",
    "She opened a new bank account.",
    "We sat on the bank and watched the water."
]


for sentence in ambiguous_sentences:

    meaning, financial_score, river_score = \
        resolve_bank_ambiguity(sentence)

    print("Sentence:", sentence)
    print("Financial score:", financial_score)
    print("River score:", river_score)
    print("Resolved meaning:", meaning)


# STEP 15: PREPARE DATA FOR N-GRAM LANGUAGE MODELS
# a statistical language model that predicts the prob of the next word in the sequence based on the previous n-1 words.

print("\nN-GRAM LANGUAGE MODEL")

# Use the lemmatized words created during preprocessing.
tokens = lemmatized_words


print("Total tokens available for language modeling:",
      len(tokens))


# Train-Test Split

split_index = int(len(tokens) * 0.8) #with 0.8 because it sets aside 80% data for training

train_tokens = tokens[:split_index]
test_tokens = tokens[split_index:]


print("Training tokens:", len(train_tokens))
print("Testing tokens:", len(test_tokens))


# STEP 16: CREATE N-GRAM COUNTS

from collections import Counter


def create_ngrams(tokens, n):
    """
    Create n-grams from a list of tokens.
    """

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


# Create unigram, bigram and trigram lists

unigrams = create_ngrams(train_tokens, 1)
bigrams = create_ngrams(train_tokens, 2)
trigrams = create_ngrams(train_tokens, 3)


# Count n-grams

unigram_counts = Counter(unigrams)
bigram_counts = Counter(bigrams)
trigram_counts = Counter(trigrams)


print("\nNumber of unique unigrams:",
      len(unigram_counts))

print("Number of unique bigrams:",
      len(bigram_counts))

print("Number of unique trigrams:",
      len(trigram_counts))


# STEP 17: N-GRAM PROBABILITIES

def unigram_probability(word):
    """
    Calculate unigram probability using Laplace smoothing.
    """
    #laplace smoothing is a technique used in statistics and machine learning to fix the problem of zero probability
    vocabulary_size = len(set(train_tokens))

    count = unigram_counts.get((word,), 0)

    total_words = len(train_tokens)

    probability = (
        (count + 1) /
        (total_words + vocabulary_size)
    )

    return probability


def bigram_probability(word1, word2):
    """
    Calculate smoothed bigram probability.
    """

    vocabulary_size = len(set(train_tokens))

    bigram_count = bigram_counts.get(
        (word1, word2), 0
    )

    previous_count = unigram_counts.get(
        (word1,), 0
    )

    probability = (
        (bigram_count + 1) /
        (previous_count + vocabulary_size)
    )

    return probability


def trigram_probability(word1, word2, word3):
    """
    Calculate smoothed trigram probability.
    """

    vocabulary_size = len(set(train_tokens))

    trigram_count = trigram_counts.get(
        (word1, word2, word3), 0
    )

    previous_count = bigram_counts.get(
        (word1, word2), 0
    )

    probability = (
        (trigram_count + 1) /
        (previous_count + vocabulary_size)
    )

    return probability


# STEP 18: DISPLAY EXAMPLE N-GRAM PROBABILITIES

print("\nEXAMPLE N-GRAM PROBABILITIES")


example_word = train_tokens[0]

print(
    f"P({example_word}) =",
    unigram_probability(example_word)
)


if len(train_tokens) >= 2:

    word1 = train_tokens[0]
    word2 = train_tokens[1]

    print(
        f"P({word2} | {word1}) =",
        bigram_probability(word1, word2)
    )


if len(train_tokens) >= 3:

    word1 = train_tokens[0]
    word2 = train_tokens[1]
    word3 = train_tokens[2]

    print(
        f"P({word3} | {word1}, {word2}) =",
        trigram_probability(
            word1,
            word2,
            word3
        )
    )


# STEP 19: SENTENCE PROBABILITY

import math


def unigram_sentence_probability(sentence):
    """
    Calculate probability of a sentence
    using the unigram model.
    """

    probability = 1.0

    for word in sentence:

        probability *= unigram_probability(word)

    return probability


def bigram_sentence_probability(sentence):
    """
    Calculate probability using the bigram model.
    """

    if len(sentence) < 2:
        return 0

    probability = 1.0

    for i in range(1, len(sentence)):

        probability *= bigram_probability(
            sentence[i - 1],
            sentence[i]
        )

    return probability


def trigram_sentence_probability(sentence):
    """
    Calculate probability using the trigram model.
    """

    if len(sentence) < 3:
        return 0

    probability = 1.0

    for i in range(2, len(sentence)):

        probability *= trigram_probability(
            sentence[i - 2],
            sentence[i - 1],
            sentence[i]
        )

    return probability


# STEP 20: PERPLEXITY

def calculate_unigram_perplexity(test_data):
    """
    Calculate unigram perplexity.
    """

    log_probability = 0

    for word in test_data:

        probability = unigram_probability(word)

        log_probability += math.log(probability)

    perplexity = math.exp(
        -log_probability / len(test_data)
    )

    return perplexity


def calculate_bigram_perplexity(test_data):
    """
    Calculate bigram perplexity.
    """

    if len(test_data) < 2:
        return float("inf")

    log_probability = 0
    count = 0

    for i in range(1, len(test_data)):

        probability = bigram_probability(
            test_data[i - 1],
            test_data[i]
        )

        log_probability += math.log(probability)

        count += 1

    perplexity = math.exp(
        -log_probability / count
    )

    return perplexity


def calculate_trigram_perplexity(test_data):
    """
    Calculate trigram perplexity.
    """

    if len(test_data) < 3:
        return float("inf")

    log_probability = 0
    count = 0

    for i in range(2, len(test_data)):

        probability = trigram_probability(
            test_data[i - 2],
            test_data[i - 1],
            test_data[i]
        )

        log_probability += math.log(probability)

        count += 1

    perplexity = math.exp(
        -log_probability / count
    )

    return perplexity


# Calculate perplexity

unigram_perplexity = calculate_unigram_perplexity(
    test_tokens
)

bigram_perplexity = calculate_bigram_perplexity(
    test_tokens
)

trigram_perplexity = calculate_trigram_perplexity(
    test_tokens
)


print("\nN-GRAM MODEL EVALUATION")

print(
    f"Unigram Perplexity : {unigram_perplexity:.4f}"
)

print(
    f"Bigram Perplexity  : {bigram_perplexity:.4f}"
)

print(
    f"Trigram Perplexity : {trigram_perplexity:.4f}"
)


# STEP 21: NEXT-WORD PREDICTION USING N-GRAMS

def predict_next_word_bigram(previous_word):

    candidates = {}

    for (word1, word2), count in bigram_counts.items():

        if word1 == previous_word:

            candidates[word2] = count

    if not candidates:
        return None

    return max(
        candidates,
        key=candidates.get
    )


def predict_next_word_trigram(word1, word2):

    candidates = {}

    for (w1, w2, w3), count in trigram_counts.items():

        if w1 == word1 and w2 == word2:

            candidates[w3] = count

    if not candidates:
        return None

    return max(
        candidates,
        key=candidates.get
    )


print("\nNEXT-WORD PREDICTION")


if len(train_tokens) >= 2:

    previous_word = train_tokens[0]

    prediction = predict_next_word_bigram(
        previous_word
    )

    print(
        f"Bigram prediction after "
        f"'{previous_word}': {prediction}"
    )


if len(train_tokens) >= 3:

    word1 = train_tokens[0]
    word2 = train_tokens[1]

    prediction = predict_next_word_trigram(
        word1,
        word2
    )

    print(
        f"Trigram prediction after "
        f"'{word1} {word2}': {prediction}"
    )


# STEP 22: NEURAL LANGUAGE MODEL

print("\nNEURAL LANGUAGE MODEL")


# TensorFlow / Keras

import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Prepare complete text

neural_text = " ".join(tokens)


# Create tokenizer

tokenizer = Tokenizer(
    oov_token="<OOV>"
)

tokenizer.fit_on_texts([neural_text])


word_index = tokenizer.word_index

vocab_size = len(word_index) + 1


print("\nNeural vocabulary size:", vocab_size)


# Create input sequences

sequences = []

for sentence in sentences:

    sentence = sentence.lower()

    sequence = tokenizer.texts_to_sequences(
        [sentence]
    )[0]

    # Need at least two words
    if len(sequence) >= 2:

        for i in range(1, len(sequence)):

            ngram_sequence = sequence[:i + 1]

            sequences.append(ngram_sequence)


# Pad sequences

max_sequence_length = max(
    len(sequence)
    for sequence in sequences
)


padded_sequences = pad_sequences(
    sequences,
    maxlen=max_sequence_length,
    padding="pre"
)


# Separate input and output

X = padded_sequences[:, :-1]

y = padded_sequences[:, -1]


print("Training sequences:", len(X))
print("Sequence length:", X.shape[1])
print("\n")


# STEP 23: BUILD LSTM MODEL

from tensorflow.keras import Input

model = Sequential([
    Input(shape=(X.shape[1],)),

    Embedding(
        input_dim=vocab_size,
        output_dim=64
    ),

    LSTM(64),

    Dense(
        vocab_size,
        activation="softmax"
    )
])


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


print("\nNeural network architecture:")
model.summary()


# STEP 24: TRAIN NEURAL LANGUAGE MODEL

early_stopping = EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True
)


history = model.fit(
    X,
    y,
    epochs=100,
    batch_size=8,
    verbose=1,
    callbacks=[early_stopping]
)


# STEP 25: NEURAL MODEL EVALUATION

loss, accuracy = model.evaluate(
    X,
    y,
    verbose=0
)


neural_perplexity = math.exp(loss)


print("\nNEURAL LANGUAGE MODEL RESULTS")

print(f"Training Loss     : {loss:.4f}")
print(f"Training Accuracy : {accuracy:.4f}")
print(f"Neural Perplexity : {neural_perplexity:.4f}")


# STEP 26: NEURAL NEXT-WORD PREDICTION

reverse_word_index = {
    index: word
    for word, index in word_index.items()
}


def predict_next_word(seed_text):

    sequence = tokenizer.texts_to_sequences(
        [seed_text.lower()]
    )[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=X.shape[1],
        padding="pre"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_word = reverse_word_index.get(
        predicted_index,
        "<UNKNOWN>"
    )

    return predicted_word


print("\nNEURAL NEXT-WORD PREDICTION")


test_phrases = [
    "natural language",
    "machine learning",
    "language model"
]


for phrase in test_phrases:

    prediction = predict_next_word(
        phrase
    )

    print(
        f"Input: '{phrase}'"
    )

    print(
        f"Predicted next word: '{prediction}'\n"
    )


# STEP 27: MODEL COMPARISON

print("STATISTICAL VS NEURAL MODEL COMPARISON")


print(
    f"\n{'Model':<20}"
    f"{'Perplexity':<20}"
)

print("-" * 40)

print(
    f"{'Unigram':<20}"
    f"{unigram_perplexity:<20.4f}"
)

print(
    f"{'Bigram':<20}"
    f"{bigram_perplexity:<20.4f}"
)

print(
    f"{'Trigram':<20}"
    f"{trigram_perplexity:<20.4f}"
)

print(
    f"{'Neural LSTM':<20}"
    f"{neural_perplexity:<20.4f}"
)


# STEP 28: CREATE COMPARISON TABLE

import pandas as pd


comparison = pd.DataFrame({
    "Model": [
        "Unigram",
        "Bigram",
        "Trigram",
        "Neural LSTM"
    ],

    "Perplexity": [
        unigram_perplexity,
        bigram_perplexity,
        trigram_perplexity,
        neural_perplexity
    ]
})

print("\n")
print(comparison)


# STEP 29: VISUALIZE PERPLEXITY

import matplotlib.pyplot as plt


plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["Perplexity"]
)

plt.xlabel("Language Model")
plt.ylabel("Perplexity")
plt.title(
    "Comparison of Statistical and Neural Language Models"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# STEP 30: PLOT NEURAL MODEL TRAINING LOSS

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"]
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.title(
    "Neural Language Model Training Loss"
)

plt.tight_layout()

plt.show()


# STEP 31: SAVE RESULTS

results_file = os.path.join(
    project_folder,
    "model_comparison.csv"
)


comparison.to_csv(
    results_file,
    index=False
)


print("\nResults saved to:")
print(results_file)

print("\nNLP PROJECT COMPLETED")