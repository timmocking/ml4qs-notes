# Lecture 3: Feature engineering

Feature engineering for time series data can occur in the time domain, the frequency domain. Additionally, specific features can be constructed based on the sensors used.

## Temporal domain

### Numerical features

By summarizing values of a numerical attribute over certain window size λ, we can incorporate information about previous instances. Of course this assumes a temporal ordering of the data.

This information can be used to compute new feature based on the window such as:

* Mean
* Median
* Standard deviation
* Minimum/Maximum

The considerations for taking different λ sizes are similar to that as we've seen with granularities in lecture 1.



### Categorical features

Summarizing categorical features is a bit more complicated than numerical features.

The following approach is described by Batal *et al.* (2013):

Two different relationships can be distinguished over again a window size λ :

* Succession (b)
  * One categorical feature happened before another one
* Co-occurence (c)
  * Different categorical features co-occur (from different attributes)

We can consider the occurence of different patterns such as one activity occuring before another, or a feature such as heart rate co-occuring with a feature such as running.

For these patterns we can define **support** as follows:

[FORMULA]

We only want to focus on patterns with sufficient support. 



## Frequency domain

Data with periodicity (e.g. walking, running) contain temporal patterns for which we can extract features related to the frequency.



### Fourier transformation

The Fourier transformation can be used to identify which frequencies we observe within λ.

Time series data can consist of different temporal patterns added together. Thus, we can decompose this into the original frequencies.

This is done by creating sinusoid functions with different frequencies.  We can then figure out how these different frequencies account for the values we measure by looking at the amplitudes they result in.

We can find these best values using Fast Fourier Transform.

[An intuitive explanation of the Fourier transformation by 3Blue1Brown](https://www.youtube.com/watch?v=spUNpyF58BY). 

After performing the Fourier transformation, we can calculate additional new features:

* Highest amplitude frequency

  

* Frequency weighted signal average

  

* Power spectrum entropy

  

## Feature engineering for text data

We can decompose a sentence (and reduce noise) using the following steps:

* Tokenization (split sentence into different words)
* Lower case (change uppercase letters to lowercase)
* Stemming (identify stem for all words)
  * This means that for example "walking" and "walked" become "walk"
* Stop word removal



**Bag of Words (BOW)**

We can occurrences of different n-grams within texts.

Consider the following sentences:

sent1: "I like trains"

sent2: "I like ice-cream"

"sent3: Ice-cream, I like"

We could create the following bag of words using 1-grams as follows:

| Sentence | I    | like | trains | ice-cream |
| -------- | ---- | ---- | ------ | --------- |
| sent1    | 1    | 1    | 1      | 0         |
| sent2    | 1    | 1    | 1      | 1         |
| sent3    | 1    | 1    | 0      | 1         |

And for 2-grams (bigrams):

| Sentence | [I, like] | [like, trains] | [like, ice-cream] | [Ice-cream, I] |
| -------- | --------- | -------------- | ----------------- | -------------- |
| sent1    | 1         | 1              | 0                 | 0              |
| sent2    | 1         | 0              | 0                 | 0              |
| sent3    | 1         | 0              | 0                 | 1              |

Larger n-grams can contain more information but require more data to be informative and are also more computationally costly.

A main drawback for BOW is that it does not take word-uniqueness into account. 



**TF-IDF**

TF-IDF effectively scales the BOW by the number of times a certain n-gram occurs in a corpus (a collection of sentences).

[FORMULA]

This ensures that very frequent (most likely not predictive) words don't become too dominant. 

  

**Topic modelling**

In topic modelling, we assume that a certain corpus contains k topics. 

Topics are associated with certain words, which carry weights for all the k topics.

In order to find topics, we can use Latent Dirichlet Allocation (LDA). LDA assumes that words come form a Poisson distribution with a distribution over topics (a Dirichlet distribution). 