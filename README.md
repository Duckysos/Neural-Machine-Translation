# Neural Machine Translation (NMT)

This repository contains a simple neural machine translation (NMT) system implemented using PyTorch and a custom sequence‑to‑sequence library.  The primary work is contained in the Jupyter notebook **`fnlp_assignment2_v2.ipynb`**, which demonstrates how to build, train and evaluate a translation model from scratch.  The notebook uses a bilingual dataset to learn to translate from a source language (e.g., English) into a target language (e.g., French) and showcases common practices such as data preprocessing, building vocabulary, defining the encoder–decoder architecture, training with teacher forcing and evaluating with the BLEU score.  An earlier version of the notebook (`fnlp_assignment2_v1.ipynb`) is also present but is not the focus of this README.

## Repository structure

| Item | Description |
|---|---|
|`fnlp_assignment2_v2.ipynb`|Main notebook containing code and explanations for the NMT assignment.  It loads a dataset, builds vocabularies, defines encoder and decoder networks, trains the model and evaluates translations. |
|`Seq2Seq/`|Python package with reusable modules used by the notebook.  It contains classes for the encoder–decoder model (`model.py`), helper functions for the Transformer architecture (`transformer_utils.py`) and utilities for vocabulary handling (`vocab.py`). |
|`requirements.txt`|Lists the Python packages required to run the notebook, including `torch`, `torchtext`, `numpy`, and `matplotlib`.  Installing these ensures that the notebook runs without missing dependencies. |
|`README.md`|You are reading it. The original version of this file was a short description; this updated README provides a detailed explanation of the notebook and how to run it. |

## Overview of the notebook

The notebook is structured in logical sections that guide you through the process of building a neural machine translation system.  A high‑level summary of the key parts is given below.

### 1. Data loading and preprocessing

1. **Dataset download:** The notebook begins by downloading a bilingual dataset (such as a subset of the Multi30k or IWSLT datasets).  It uses the `torchtext` library to fetch parallel sentence pairs for the source and target languages.
2. **Tokenization and vocabulary:** Each sentence is tokenized into word tokens using simple whitespace tokenization or `spaCy` tokenizers.  The notebook builds a vocabulary for each language, mapping words to integer indices and adding special tokens (e.g., `<sos>`, `<eos>` and `<pad>`).
3. **Sequence batching:** Sentence pairs are batched and padded to the same length using PyTorch’s `DataLoader`.  Padding ensures that variable‑length sequences can be processed in parallel by the model.

### 2. Model architecture

The notebook implements a classic encoder–decoder architecture with attention for neural machine translation:

* **Encoder:** A recurrent neural network (RNN) encoder reads the source sentence and encodes it into a sequence of hidden states.  Typically this is implemented as a stack of gated recurrent units (GRUs) or long short‑term memory (LSTM) layers.
* **Decoder:** Another RNN decoder generates the target sentence token by token, conditioned on the encoder’s context vector and previous outputs.  Teacher forcing is used during training to feed the ground‑truth token at each step.
* **Attention mechanism:** To allow the decoder to focus on different parts of the source sentence at each time step, the notebook employs an attention module.  The attention weights determine which encoder hidden states contribute most to the current decoding step.

The implementation makes use of the reusable classes defined in the `Seq2Seq` module.  For example, the `model.py` file defines `EncoderRNN`, `DecoderRNN` and `Seq2Seq` classes, while `transformer_utils.py` contains helper functions for positional encodings and multi‑head attention when experimenting with Transformer architectures.

### 3. Training and optimization

1. **Loss function:** The notebook uses cross‑entropy loss on the decoder outputs, ignoring loss contributions from padded tokens.
2. **Optimizer and learning rate:** An optimizer such as Adam or SGD with momentum is configured.  The learning rate is set according to typical NMT training practices and may include scheduling (e.g., step decay or warm‑up).
3. **Teacher forcing:** During each training step, the decoder receives the correct previous token (with some probability) to stabilize learning.  Teacher forcing ratio is a hyperparameter that can be tuned.
4. **Training loop:** The notebook iterates over batches of sentence pairs, computes the forward pass through the encoder and decoder, calculates loss, performs backpropagation and updates model weights.  Periodic progress logging (such as printing training loss after certain epochs) helps monitor convergence.

### 4. Evaluation

After training, the notebook evaluates the model on held‑out validation data:

* **Translation generation:** For each source sentence, the notebook uses the trained model to generate a target translation by greedily selecting the most probable token at each step (or optionally using beam search).
* **BLEU score:** The quality of translations is measured with the BLEU (Bilingual Evaluation Understudy) score, which compares the generated translations against reference translations.  Higher BLEU scores indicate better translation quality.
* **Qualitative examples:** The notebook prints sample source sentences and their model translations alongside the ground‑truth target sentences.  This helps visually inspect the model’s performance and common error patterns (e.g., under‑translation or misordered words).

### 5. Results and observations

The notebook reports training and validation losses across epochs and shows BLEU scores on the validation set.  It may also provide plots of loss curves or attention heatmaps to illustrate where the model focuses when translating.  Observations typically include:

* The model learns to produce reasonable translations after several epochs, with training loss decreasing and BLEU scores improving.
* Attention heatmaps highlight how the decoder attends to different source tokens when generating each target word.
* Common limitations include sensitivity to rare words, difficulties translating long sentences and occasional grammatical errors, which can be addressed by larger models, more data or transformer architectures.

