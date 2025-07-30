# N-Gram Language Model and Spelling Corrector from Scratch

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project features a from-scratch implementation of an **N-Gram Language Model** in Python. The model is trained on a corpus of news articles and is then used to build a practical **context-aware spelling corrector**.

---

## 📋 Project Workflow

The project is divided into two main parts: building the language model and then using it for spelling correction.

### 1. N-Gram Language Model
The language model is implemented from scratch using only standard Python libraries. Its key features include:

* **Data Preprocessing:** A standard NLP pipeline is used to clean and prepare the text corpus, including sentence segmentation, word tokenization, case folding, punctuation removal, and Porter stemming.
* **Vocabulary Management:** A vocabulary is built from the training data. Words with low frequency are replaced with an `<UNK>` token to handle out-of-vocabulary words.
* **Probability Estimation:** N-gram probabilities are calculated using Maximum Likelihood Estimation (MLE) with **add-k (Laplace) smoothing** to handle unseen n-grams.
* **Evaluation:** The model's performance is measured using **perplexity** on a held-out test set.
* **Text Generation:** The model can generate new sentences by sampling from the learned probability distributions.

### 2. Spelling Correction Application
The trained n-gram model serves as the core of a context-aware spelling corrector. The process is as follows:

1.  For a potentially misspelled word in a sentence, a set of **candidate corrections** is generated (e.g., words with a low edit distance).
2.  The language model then calculates the probability of the sentence for each candidate word placed in its original context.
3.  The candidate that yields the highest sentence probability is chosen as the most likely correction.
4.  The performance of the spelling corrector is evaluated using **Word Error Rate (WER)** and **Character Error Rate (CER)**.

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* NLTK (for tokenization and stemming)

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Open and run the cells in the main Jupyter Notebook. The notebook will call the necessary functions from the `.py` scripts to perform data preparation, train the language model, and run the spelling corrector.

---

## 📄 License
This project is licensed under the MIT License.