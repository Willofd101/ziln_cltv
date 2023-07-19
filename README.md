# ZILN_CLTV

A python package to train & evaluate Customer Lifetime Value(CLTV) models using Neural Networks & [ZILN loss](https://github.com/google/lifetime_value).

## Paper
Wang, Xiaojing, Liu, Tianqi, and Miao, Jingang. (2019). A Deep Probabilistic Model for Customer Lifetime Value Prediction : [arXiv:1912.07753](https://arxiv.org/abs/1912.07753).

## Installation
 Run the below command in command prompt(cmd) to install directly:

```
pip install git+https://github.com/djdeepak72/ziln_cltv.git
```

OR

 
Clone the directory, `cd` into it & run the below:

```
pip install .
```


## What is this package used for?

`ziln_cltv` is a user-friendly packaged form of research conducted by google to train neural network models to predict Customer Lifetime Value. What makes this research special is the use of zero-inflated lognormal (ziln) loss in building CLTV models where the distribution of LTV is heavily-tailed. Usually majority of customers donot come back to buy products which causes your dataset to have many customers with zero ltv. Metrics like mean squared error cannot account for such skewed distribution because of these one-time customers, hence making models very volatile.

ZILN distribution models LTV as mix of zero point mass & lognormal distribution. This lets you model both the skewed nature of LTV as well as the churn probability associated with it, giving you easy uncertainity quantification of your point prediction. 

## What data do you need to build CLTV models using this package?

Google's research indicates that predicting LTV beyond a 1 year timeline can give you less accurate predictions. 
So ideally you would need to build CLTV models to predict the lifetime value of the customer in the 1 year following the customer's first purchase. To do that you would need:
* Total purchase amount by the customer on their first date of purchase.
* **Target variable**: Total purchase amount by the customer in the year following the customer's first date of purchase i.e Total purchase amount between day 2 to day 365. You would be fitting the model with this as the response. Alternatively, if you want to model LTV with a wider timeline, say LTV for more than 1 year, you can still do so. But remember that model performance tends to deteriorate especially due to changing dynamic of customers , markets, prices, inflation etc. Similarly, you can also use this package to model LTV for next 3 months or 6 or 9 etc.
* Lastly you'd need other features that can improve your model like, product details of customer's first purchase, customer characteristics like location, age, payment type & other things you have on the customer.

## How do I use this package?

You can learn more about the package by following the notebook in the [tutorial](tutorial) folder which uses [Kaggle Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) LTV competition dataset & shows you how to build & train your CLTV model & evaluate performance of both LTV & Churn predictions.

