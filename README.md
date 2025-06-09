# AI Algorithms From Scratch

This repository collects simple implementations of algorithms that serve as a learning roadmap for anyone starting a career in **Artificial Intelligence Engineering**. Each file demonstrates a fundamental concept or common technique in machine learning or natural language processing.

The code is designed for students curious about how models are built from the ground up, beginning with gradient descent and ending with a simplified GPT example.

## Structure

- **01.gradient-descent.py** — demonstrates function optimization using gradient descent.
- **02.linear-regression-foward.py** — computes predictions and errors for a linear regression model.
- **03.linear-regression-training.py** — trains a linear regression model with manual weight updates.
- **04.py-torch-basics.py** — introductory examples of tensor manipulation in PyTorch.
- **05.neural-network.py** — a simple neural network with linear layers.
- **06.digit-classifier.py** — digit classifier (MNIST) and inference on an image (`digit.png`).
- **07.intro-NLP.py** — prepares text and encodes tensors for NLP tasks.
- **08.sentiment-analysis-and-word-embeddings.py** — sentiment analysis model using embeddings.
- **09.gpt-dataset-example.py** — creates context/target pairs for training GPT-style models.
- **10.self-attention.py** — basic self-attention mechanism with triangular mask.
- **11.multi-headed-attention.py** — multihead version of the attention mechanism.
- **12.transformer-block.py** — full Transformer block with attention and feed-forward layers.
- **13.GPT.py** — simplified GPT model with embeddings, Transformer blocks and final projection.
- **14.gpt-answers.py** — generates text using a trained GPT.
- **extras/rnn.py** — additional recurrent neural network (RNN) example.
- **fundamentals_ai_articles/** — a chronological collection of landmark AI papers.
- **Notebooks** (`GPT_FROM_SCRATCH.ipynb`, `fine-tuning-gemma.ipynb`, `sentiment_analysis_model.ipynb`) — interactive explorations of concepts shown in the code.

## Why the `fundamentals_ai_articles` folder matters

Inside this folder you will find a timeline of pivotal publications that shaped the field of AI, from early neural network research all the way to state-of-the-art Large Language Models (LLMs). Reading these papers is strongly recommended for engineers who want to deeply understand how modern LLMs came to be.

## How to use

1. Install Python 3 and create a virtual environment (optional):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the main dependencies:
   ```bash
   pip install torch torchvision torchtyping
   ```
3. Run the desired script:
   ```bash
   python 06.digit-classifier.py
   ```
   Feel free to explore each file individually and tweak the hyperparameters to experiment with different behaviors.

## Contributing

This repository is meant for learning. If you find an issue or have suggestions for improvement, open an issue or submit a pull request.

Happy learning!
