import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from typing import List

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence

loaded_model = load_model("my_model.h5")


word_to_index = imdb.get_word_index()
MAXLEN = 250


def encode_text(raw_text: str) -> List[int]:  # raw text -> tokenize -> encode -> pad
    """
    Encodes raw text into a sequence of integers, padding it to a fixed length.
    :param raw_text: The input text to encode.
    :return: A padded sequence of integers representing the encoded text.
    """
    tokens = text_to_word_sequence(raw_text)
    encoded_text = [
        word_to_index[token] if token in word_to_index else 0 for token in tokens
    ]
    return sequence.pad_sequences([encoded_text], maxlen=MAXLEN)[0]


index_to_word = {index: word for word, index in word_to_index.items()}


def decode_text(encoded_text: List[int]) -> str:
    """
    Decodes a sequence of integers back into a human-readable text.
    :param encoded_text: The input sequence of integers to decode.
    :return: A string representing the decoded text.
    """
    text = ""
    pad = 0
    for num in encoded_text:
        if num != pad:
            text += index_to_word[num] + " "

    return text[:-1]


def predict(raw_text: str, model) -> None:
    """
    Predicts the sentiment of a given text using the loaded model.
    :param raw_text: The input text to analyze.
    :param model: The pre-trained sentiment analysis model.
    :return: None, prints the prediction result.
    """
    # prepare input
    encoded_text = encode_text(raw_text)  # encoding
    input_tensor = np.zeros((1, MAXLEN))
    input_tensor[0] = encoded_text  # add batch dimension for model compatability
    # predict
    prediction = model.predict(input_tensor)
    threshold = 0.5
    predicted_value = prediction.item()
    if predicted_value < threshold:
        print(f"The review is negative ⛔️, the predicted value: {predicted_value:.2f}")
    elif predicted_value > threshold:
        print(f"The review is positive ✅, the predicted value: {predicted_value:.2f}")
    else:
        print(f"The review is neutral 😑, the predicted value: {predicted_value:.2f}")


# example usage
if __name__ == "__main__":
    positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
    prediction1 = predict(positive_review, loaded_model)
    negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
    prediction2 = predict(negative_review, loaded_model)
