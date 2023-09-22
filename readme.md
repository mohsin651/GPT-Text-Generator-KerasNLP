# GPT Text Generation with KerasNLP

## Description

This project demonstrates how to use KerasNLP to train a mini-GPT (Generative Pre-trained Transformer) model for text generation. GPT is a powerful language model that can generate coherent and contextually relevant text given a prompt.

We trained the model on the simplebooks-92 corpus, a dataset created from several novels. This dataset is suitable for our example due to its small vocabulary and high word frequency, making it ideal for training a model with limited parameters.

In this project, covered is:

- Data setup and preprocessing
- Tokenization using KerasNLP
- Building a mini-GPT model
- Training the model
- Text generation using various techniques

**Note:** If you're running this project on Google Colab, consider enabling GPU runtime for faster training (Not nessecarily below 10 epochs).

## Setup

Before starting, make sure you have KerasNLP installed. You can install it with the following command:

```bash
pip install keras-nlp
````
## Data Loading
We downloaded the simplebooks-92 dataset, which consists of 1,573 Gutenberg books. This dataset has a small vocabulary and is suitable for our task. We will load and preprocess the training and validation sets.

## Tokenization
We trained a tokenizer on the training dataset to create a sub-word vocabulary. Tokenization is crucial for converting text into a format that the model can understand.

## Model Building
We created a scaled-down GPT model with KerasNLP. Our model includes:
- Token and Position Embedding layer
- Multiple TransformerDecoder layers
- A final dense linear layer


We configured the model's hyperparameters such as batch size, sequence length, embedding dimensions, and more.

## Training
Trained the model using the training dataset, validating it on the validation dataset. You can adjust the number of epochs to fine-tune the model's performance.

## Text Generation
Explore different text generation techniques using our trained model:

- Greedy Search: Selecting the most probable token at each step.
- Beam Search: Considering multiple probable sequences to reduce repetition.
- Random Search: Sampling tokens based on softmax probabilities.
- Top-K Search: Sampling from the top-K most probable tokens.
- Top-P (Nucleus) Search: Sampling based on a dynamic probability threshold.

Demonstrated how to generate text using these methods and showcase their advantages and limitations.

## Conclusion
This project provides a comprehensive guide to training a mini-GPT model for text generation using KerasNLP. It covers data preprocessing, model architecture, training, and various text generation techniques. To further understand Transformers and explore training full-sized GPT models, you can refer to additional resources.



