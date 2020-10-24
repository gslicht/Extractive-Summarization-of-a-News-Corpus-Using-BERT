# Extractive-Summarization-of-a-News-Corpus-Using-BERT

This project takes two approaches to modelling ectractive summarization of a generalised news corpus using BERT sentence embedding. The first focuses on learning the internal embedding structure of each article and relates closest to our intuition that a good summarizer is one that is able to parse meaning and classify the salient points of a document. The supervised models considered here limit the features set to the sentence embeddings and their derivatives. The results are then compared to an unsupervised baseline using TextRank. The other less generalizable approach takes advantage of the known ordering structure particular to news articles and exploited in LEDE3 (pick first three sentences as the summary). We add sequential information to the embeddings and compare the results to LEDE3 as the baseline.

Both supervised approaches perform better than the baselines with the best results coming from a simple elastic net for the TextRank baseline (Rouge1 F1: 41.5% vs 34%) and from a 50 neuron bidirectional LSTM for the LEDE3 baseline (Roug1 F1: 60.3% vs 56.7%).  

The Cornell Newsroom database was used with articles filtered for extractive summaries and truncated to 5,000 articles representing over 166,000 sentences. 

The repository has the following components:

1. requirements.txt
2. Codebase
  .1 Data Wrangling
  .2 Data Exploration
  .3 Machine Learning Implementation
  .4 Results 
3. Final report (pdf)
