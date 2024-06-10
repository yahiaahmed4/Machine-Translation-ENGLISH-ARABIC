# Machine Translation ENGLISH-ARABIC

This repository contains two notebooks for training Sequence-to-Sequence (Seq2Seq) Recurrent Neural Network (RNN) models for the task of Machine Translation on an English-Arabic dataset. The two notebooks are as follows:

1. **Seq2Seq RNN Model without Attention**
2. **Seq2Seq RNN Model with Attention**

## Dataset

The dataset used for this project is a parallel corpus of English and Arabic sentences. The data is preprocessed to remove English characters and diacritics (Tashkeel) from the Arabic sentences.

## Requirements

The notebooks are designed to run in a Google Colab environment. To run them locally or on another platform, you will need to install the following dependencies:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Google Colab (for accessing Google Drive)

You can install the necessary libraries using the following commands:

```sh
pip install tensorflow pandas numpy matplotlib
```

## Project Structure

The project consists of two main notebooks:

1. **Seq2Seq RNN Model without Attention (encoder_decoder_without_attention.ipynb)**
2. **Seq2Seq RNN Model with Attention (encoder_decoder_with_attention.ipynb)**

### Common Steps

Both notebooks share some common steps for data loading, preprocessing, and model training:

1. **Data Loading and Preprocessing**
    - Load the dataset from a file stored in Google Drive.
    - Preprocess the data by removing English characters and diacritics (Tashkeel) from the Arabic sentences.
    - Tokenize and pad the sentences to create input and output sequences.

2. **Model Training**
    - Define the encoder and decoder models using Bidirectional LSTMs.
    - Compile and train the model with categorical cross-entropy loss and Adam optimizer.
    - Use early stopping to prevent overfitting.

### Seq2Seq RNN Model without Attention

In the notebook `encoder_decoder_without_attention.ipynb`, a Seq2Seq model is implemented without using an attention mechanism. The main components are:

- **Encoder:** A Bidirectional LSTM that processes the input sequence and returns the final states.
- **Decoder:** An LSTM that takes the encoder states as initial states and generates the output sequence.
- **Dense Layer:** A fully connected layer with softmax activation to generate the final output.

### Seq2Seq RNN Model with Attention

In the notebook `encoder_decoder_with_attention.ipynb`, an attention mechanism is added to the Seq2Seq model. The main changes are:

- **Attention Layer:** An attention mechanism that calculates the context vector by attending to the encoder outputs.
- **Context Vector:** Concatenated with the decoder outputs before passing through the dense layer.

## Training Results

### Seq2Seq RNN Model without Attention

- **Training Loss:** 0.4315
- **Training Accuracy:** 0.8796
- **Validation Loss:** 0.8344
- **Validation Accuracy:** 0.7712

### Seq2Seq RNN Model with Attention

- **Training Loss:** 0.4359
- **Training Accuracy:** 0.8779
- **Validation Loss:** 0.8352
- **Validation Accuracy:** 0.7711


## Conclusion

This project demonstrates the implementation of Seq2Seq RNN models for machine translation on an English-Arabic dataset. The addition of the attention mechanism helps improve the model's performance by allowing the decoder to focus on relevant parts of the input sequence during translation.

Feel free to explore the notebooks and modify them for further experimentation and improvements.



