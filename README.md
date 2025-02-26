# Assignment 1: N-gram Language Model

This repository contains the implementation of an n-gram language model for our course assignment. The project focuses on building unsmoothed and smoothed unigram and bigram models, handling unknown words, and computing perplexity on a validation set.

## Features

- **Unigram and Bigram Probability Computation:**  
  Calculate probabilities based on token counts from a training corpus.
  
- **Smoothing Techniques:**  
  Implement both Laplace (add-1) smoothing and add-\(k\) smoothing to handle data sparsity.
  
- **Unknown Word Handling:**  
  Replace low-frequency words with a special `<UNK>` token to manage unseen words.
  
- **Perplexity Calculation:**  
  Compute the perplexity for both unigram and bigram models using smoothed probabilities.
  
- **Experimental Analysis:**  
  Compare different smoothing constants and minimum count thresholds, and visualize the effects using graphs.

## Code File

- `Assignment1-code.ipynb` – Main script implementing the language model, smoothing, and perplexity calculations.

##Team Members
	•	Farhan Jamil (FXJ200003)
	•	Shoaib Huq (SXH200053)
	•	Lerich Osay (LRO200000)
	•	Sriram Sendhil (SXS200327)
