# Lecture 4: Clustering

## Learning setup

There are different ways in which we can apply machine learning on a quantified self dataset.

**Per instance**

In this case, we apply clustering on all instances, ignoring possible information about possible users.

**Per person**

Self-explanatory, we apply clustering to group certain users to together.



## Distance functions

In general its important to consider **scaling**. 

We can measure different distance metrics for **per instance-level** clustering tasks as well as metrics for **dataset** level clustering tasks. 



### Distances between instances



**Euclidian distance**

Which is just the shortest path between two points, as calculated using Pythagoras. 

![](https://latex.codecogs.com/gif.latex?d%28x_%7Bi%7D%2C%20x_%7Bj%7D%29%20%3D%20%5Csqrt%7B%5Csum_%7Bk%3D1%7D%5E%7Bp%7D%28x%5E%7Bk%7D_%7Bi%7D%20-%20x%5E%7Bk%7D_%7Bj%7D%29%7D)



**Manhattan distance**

Also known as city-block distance. Consider the shortest distance, while only being able to move in the x and y-axis directions. 

![](https://latex.codecogs.com/gif.latex?d%28x_%7Bi%7D%2C%20x_%7Bj%7D%29%20%3D%20%5Csum_%7Bk%3D1%7D%5E%7Bp%7D%5Cleft%20%7C%20x%5E%7Bk%7D_%7Bi%7D%20-%20x%5E%7Bk%7D_%7Bj%7D%5Cright%20%7C)



**Minkowski distance**

Can be used to calculate both Euclidian and Manhattan, depending on value (q=1: Manhattan, q=2: Euclidian)

![](https://latex.codecogs.com/gif.latex?d%28x_%7Bi%7D%2C%20x_%7Bj%7D%29%20%3D%20%28%5Csum_%7Bk%3D1%7D%5E%7Bp%7D%5Cleft%20%7C%20x%5E%7Bk%7D_%7Bi%7D%20-%20x%5E%7Bk%7D_%7Bj%7D%5Cright%20%7C%5E%7Bq%7D%29%5E%7B%5Cfrac%7B1%7D%7Bq%7D%7D)



**Gower's similarity**

Gower's similarity is a metric which can use both dichotomous  (e.g. male/female), categorical and numerical features.

You define a similarity *s* for all points between two features:

Dichotomous attributes:  1 when both are the same, 0 otherwise

Categorical attributes: 1 if values the same, 0 otherwise

Numerical attributes: 

![](https://latex.codecogs.com/gif.latex?s%28x_%7Bi%7D%2C%20x_%7Bj%7D%29%20%3D%201%20-%20%5Cfrac%7B%5Cleft%20%7C%20s%28x%5E%7Bk%7D_%7Bi%7D%2C%20x%5E%7Bk%7D_%7Bj%7D%20%29%5Cright%20%7C%7D%7BR_%7Bk%7D%7D)

Where *R<sub>k</sub>* is the range of *k*. 



The similarity is then defined as follows where δ is 1 if values can be compared and 0 otherwise.

![](https://latex.codecogs.com/gif.latex?similarity_%7Bgowers%7D%28x_%7Bi%7D%2C%20x_%7Bj%7D%29%20%3D%20%5Cfrac%7B%5Csum_%7Bk%3D1%7D%5E%7Bp%7Ds%28x%5E%7Bk%7D_%7Bi%7D%2C%20x%5E%7Bk%7D_%7Bj%7D%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7Bp%7D%5Cdelta%28x%5E%7Bk%7D_%7Bi%7D%2C%20x%5E%7Bk%7D_%7Bj%7D%29%7D)





### Distances between datasets

For comparing **datasets** as opposed to individual instances, multiple metrics exist for datasets with or without **temporal ordering**. 

The following strategies can be followed when there is **no temporal ordering** in the dataset:

* Summarize all values per attribute into a single value and compute distances as we did per-instance level
* Estimate distributions per attribute and compare the distributions
* Compare distributions of an attribute with a statistical test

The following strategies can be followed when we have **temporal ordering**.



### Raw-data based

In the simplest case we can add up the Euclidian distances per-time point. 

Downsides to this approach is that datasets are often not of equal size, and if they are slightly shifted (e.g. one dataset starting a movement 1s later, you will get very low distances). Therefore we can use a **lag**. The **cross-correlation coefficient** between datasets is computed.



High values of this coefficient occur when two datasets overlap, which is desirable of course. The lag to use is an optimization problem. 



**Dynamic time warping**

In dynamic time-warping we make the best pairs of instances in sequences to find the minimum distance.

There are a number of requirements for pairing:

* Time order should be preserved (monotonicity condition)
  * We can go up, right or diagonal
* First and last points should be matched (boundary condition)



The algorithm

* We start at (0, 0) and move to the right (boundary condition)
* Per pair we compute the cheapest way to get there given our constraints (monotonicity condition) and the distance between the instances in the pair (e.g. euclidian distance)

Often certain bounds exist that ensure you not match very distant time points. 



### Feature-based

This is the same approach as for the non-temporal ordered data we've seen before.



### Model-based

Fit a time series model and use those parameters (again similar to non-temporal ordered, except type of model being different)



## Clustering algorithms

### K-means clustering

<img src="https://imagej.net/_images/thumb/f/fb/K-means.png/900px-K-means.png" alt="gauss_9" style="zoom:40%;" align="left"/>



In order to select the best *k* we can use **silhouette scores**. This metric considers the distance-difference between points in a cluster versus those outside the cluster.

For calculating the silhouette score *for N datapoints*:

*a(x<sub>i</sub>)* is average distance between x<sub>i</sub> and points within the cluster.

*b(x<sub>i</sub>)* is average distance between x<sub>i</sub> and points inside the closest other cluster.

The silhouette score is then defined as follows:

![](https://latex.codecogs.com/gif.latex?%5Cmathit%7Bsilhouette%7D%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cfrac%7Bb%28x_%7Bi%7D%29-a%28x_%7Bi%7D%29%7D%7Bmax%28a%28x_%7Bi%7D%29%2C%20b%28x_%7Bi%7D%29%29%7D%7D%7BN%7D)

We of course want a low *a(x<sub>i</sub>)* and a high *b(x<sub>i</sub>)*. A silhouette score of -1 is bad, a silhouette score of 1 is good. 

 We can evaluate silhouette scores for all values of k and so identify an optimal k to use for clustering.



### K-medoids clustering

K-medoids is similar to k-means clustering, but in stead of computing a centre, we use an actual point (that is most central in the cluster. 

K-medoids is more suitable to person-level clustering than k-means clustering. This is because an "average" person/dataset is not very meaningful in the case of k-means clustering.



### Hierarchical clustering

**Agglomerative clustering**

In agglomerative clustering, you start with one cluster per instance and you start merging iteratively. 



There are different criteria on which we can merge clusters:

*Single-linkage*

Calculate distance as the minimum distance between two clusters.

![](https://latex.codecogs.com/gif.latex?d%28C_%7Bk%7D%2C%20C_%7Bl%7D%29%20%3D%20%5Ctextup%7Bmin%7D%20%5C%3B%20distance%28x_%7Bi%7D%2C%20x_%7Bj%7D%29)



*Complete linkage*

Calculate distance as the maximum distance between two clusters.

![](https://latex.codecogs.com/gif.latex?d%28C_%7Bk%7D%2C%20C_%7Bl%7D%29%20%3D%20%5Ctextup%7Bmax%7D%20%5C%3B%20distance%28x_%7Bi%7D%2C%20x_%7Bj%7D%29)



*Group average*

Calculate distance as the average distance between two clusters.

![](https://latex.codecogs.com/gif.latex?d%28C_%7Bk%7D%2C%20C_%7Bl%7D%29%20%3D%20%5Cfrac%7B%5Csum_%7Bx_%7Bi%7D%7D%5Csum_%7Bx_%7Bj%7D%7D%5E%7B%7Ddistance%28x_%7Bi%7D%2C%20x_%7Bj%7D%29%7D%7B%5Cleft%20%7C%20C_%7Bk%7D%5Cright%20%7C%5Ccdot%20%5Cleft%20%7C%20C_%7Bl%7D%20%5Cright%20%7C%7D)



*Ward's criterion*

Minimizes the standard deviation. 

We look at the standard deviation within two clusters, and the standard deviation if they are merged.

![](https://latex.codecogs.com/gif.latex?d%28C_%7Bk%7D%2C%20C_%7Bl%7D%29%20%3D%20%5Csigma_%7BC_%7Bk%7D%5Ccup%20C_%7Bl%7D%7D%20-%20%5Csigma_%7BC_%7Bk%7D%7D%20-%20%5Csigma_%7BC_%7Bl%7D%7D)



**Divisive clustering**

In divisive clustering, you start with one cluster and make a split in each step.

We define a dissimilarity of a point to other points in the cluster C. 

![](https://latex.codecogs.com/gif.latex?dissimilarity%28x_%7Bi%7D%2C%20C%29%20%3D%20%5Cfrac%7Bdistance%28x_%7Bi%7D%2C%20x_j%29%7D%7B%5Cleft%20%7C%20C%20%5Cright%20%7C%7D)

We iteratively remove the most dissimilar points from C and add them to a new cluster C' until these are more dissimilar to points in C'. 

The cluster C we evaluate next is the one with the biggest diameter.



### Subspace clustering

Most clustering techniques don't handle large numbers of features. 

In the first step, we create intervals for every feature, creating a grid of so-called *units*. 

Next, we calculate the *selectivity* of a these units. Given a threshold T, we classify these as *dense* units (either 1, 0) 

Important is the concept of a *common face*, which occurs when to units share a bound in one feature (e.g. the upper bound of unit A is the lower bound of unit B) and have a similar bound in another feature. 

Units are connected in a cluster when they share a common face or when they share a unit that is a common face to both. 