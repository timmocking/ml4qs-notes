# Lecture 5: Fundamentals of supervised learning: predictive modelling without time 

## Fundamentals

“A computer program is said to learn
from experience E with respect to some class of tasks T and performance P, if its
performance at tasks in T improves with E.

In supervised learning we want to learn a functional relationship called an **unkown target function**.

This relationship is not always deterministic because there are:

* Measurement errors
* Noisy targets.

Therefore we assume an **unknown conditional target distribution**, which incorporates the uncertainty that we have in making predictions.



Consider the generating mechanism of our data *f(x)* and we assume we have a hypothesis for the target function *h(x)*.

How far is *h from *f*? This what we call **risk**, defined as *E(f, h)*

We can also compute it per point: *e(f(x), h(x))*. This is called **loss**.

Obviously this is impossible, because then we would need to know *f* , and we wouldn't need the learning process to begin with. 

We can however approximate it as follows:

![](https://latex.codecogs.com/gif.latex?E%28f%2C%20h%29%20%5Capprox%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bj%3D1%7D%5E%7BN%7De%28y_%7Bj%7D%2C%20h%28x_%7Bj%7D%29%29)

Where *e* is an means of defining the error, such as F1, AUC, MSE, etc.



We make a distinction between:

**in sample error**: the amount of mistakes you make on the training set

**out of sample error** the amount of mistakes you make on all other possible elements you have

We select the hypothesis *h* that has the lowest in-sample error on the validation set.

We are unable to learn the perfect model *f*, but for increasing amounts of data (*N*) the differences will be increasingly small.

If this is the case, then we say that a hypothesis is **PAC learnable** (probably, approximately correct). 

A hypothesis set *H* **shatters** an input set X when it can represent all possible labelling's (for a binary classification problem 2<sup>N</sup>). 

The **Vapnik-Chernovenkis (VC) dimension ** d<sub>vc</sub> of a hypothesis set *H* si the maximum number of input vectors that can be shattered.  

For the decision boundary below we shatter 16 input vectors (7 red, 9 green) so the VC dimension would be 16.

<img src="https://www.jeremyjordan.me/content/images/2017/06/Screen-Shot-2017-06-10-at-9.41.25-AM.png" alt="gauss_9" style="zoom:30%;" align="left"/>



## Predictive modelling without notion of time

### Learning setup

**Individual level** 

One model per-person. 

In case of non-temporal data, we can just use random (stratified) sampling of the data.

In case of temporal ordering, use the N time points as a test set. 



**Population level**

We can use some people as training data, and use another unseen person as test data.

We can also use unseen data of known people as test data.



### Feature selection

**Forward selection**

Identify a learning algorithm (e.g. decision tree). We fit a model on all features individually, and add the most predictive feature (e.g. lowest MSE) to a new feature set, and we refit a model on all combinations of the feature set and the remaining features.

**Backward selection**

Same as above, but we remove the least predictive features and start with fitting all features.

**Regularization**

Add a term to the error function to punish more complex models (by keeping feature weights low). 



### Learning algorithms

Not explained during the lecture. See book for explanations.