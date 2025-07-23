# Sentiment Analysis on IMDb Movie Reviews Using a trained Keras Model

## Overview

This repository provides a concise implementation of a sentiment analysis pipeline that classifies IMDb movie reviews as **positive** or **negative** using a trained deep learning model built with Keras and TensorFlow. The model predicts binary sentiment labels based on natural language input and demonstrates typical preprocessing, encoding, and inference workflows used in NLP tasks.

## Features

- Uses IMDb dataset word index for consistent tokenization.
- Converts raw text into padded integer sequences compatible with Keras models.
- Loads a previously trained and saved model (`my_model.h5`).
- Provides both encoding and decoding utilities for interpretability.
- Outputs human-readable sentiment predictions with probabilistic scores.

## Model Assumptions

* The model was trained on the **IMDb dataset** provided by Keras (`keras.datasets.imdb`).
* Input sequences are truncated or padded to a maximum length of **250 words**.
* The model outputs a single scalar between 0 and 1. The lower the number, the more negative it is predicted and the higher the number, the more positive it is predicted. 

## Code Structure

### Text Encoding

```python
def encode_text(raw_text: str) -> List[int]
```

* Tokenizes raw input text.
* Encodes tokens using the IMDb `word_to_index` mapping.
* Pads the sequence to a fixed length (250 words).

### Sentiment Prediction

```python
def predict(raw_text: str, model) -> None
```

* Encodes and formats the input for model compatibility.
* Outputs a sentiment classification with a confidence score.
* Applies a threshold of `0.5` for binary classification:

  * < 0.5 → Negative ⛔️
  * \> 0.5 → Positive ✅
  * \= 0.5 → Neutral 😑

## Example Usage

```python
if __name__ == "__main__":
    positive_review = "That movie was amazing! I really loved it and would watch it again."
    predict(positive_review, loaded_model)

    negative_review = "That movie really sucked. I hated it and wouldn't watch it again."
    predict(negative_review, loaded_model)
```

## Output Example

```bash
The review is positive ✅, the predicted value: 0.93
The review is negative ⛔️, the predicted value: 0.19
```

