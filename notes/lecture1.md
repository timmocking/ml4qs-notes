## Lecture 1: Introduction, basics of sensory data

*What is the quantified self?*

> The quantified self is any individual engaged in self-tracking of any kind of biological, physical, behavioural or environmental information. The self-tracking is driven by a certain **goal** of the individual with a desire to **act upon** the collected information.

  

*Why do people act upon the quantified self?*

* Improved health and well-being
* Entertainment



*What kind of measurements fall under the quantified self?*

* Physical activity
* Diet
* Physiological states and traits
* Mental and cognitive states and traits
* Environmental variables
* Situational variables
* Social variables



*What can we learn from the quantified self?*
Forecasting and optimizing of the variables named above using machine learning methods.



*Why is the quantified self different from other machine learning tasks?*

* Sensory noise
* Missing measurements
* Temporal data
* User interaction
* Multiple datasets



---



### Crowdsignals dataset

This dataset contains various time series measurements of users in association with activities. 

For example:

**Accelerometer** measures changes upon the phone in the x-y-z plane

**Gyroscope** measures orientation of phone compared to earth's surface

**Magnetometer** measures x-y-z orientation compared to earth's magnetic field



---



### Aggregation of time-series data

Transforming of raw time series data is usually necessary to create useable, informative features. This can be done by taking a **step size** dT and combining all measurements within this step size using a certain method (summing, averaging, etc.).

There are pros and cons of taking different step sizes.

**Increasing step size** 

* (+) Reduces amount of missing values
* (-) Possible loss of patterns with a smaller scale

**Decreasing step size**

* (+) Captures small-scale patterns more effectively
* (-) Can increase amount of missing values

Finding an ideal step size is often done using trial-and-error. Effectively its a trade-off between useful patterns vs. noise + missing values.