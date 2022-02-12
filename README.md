<h1> House Price Prediction </h1>
<p> This is an attempt at using previous housing sale data, to predict house price.
This repository contains preprocessing work - to remove bad data and create additional features.
This data is then used to train an XGB machine learning algorithm </p>
<h2> Exploratory Data Analysis </h2>
Here are a few finding that were discovered during the EDA stage.

<h2> What is XGBoost and how does it work? </h2>
<p> XGBoost stands for eXtreme Gradient Boosted trees. It 
is a popular machine learning method, which regularly wins kaggle competitions.
It is an ensemble method (takes a model and make multiple versions of it chained together).
Each tree boosts attributes that led to misclassifications of previous tree. It has various useful features:
 <ul>
  <li>Regularized boosting (prevents overfitting).</li>
  <li>Parallel Processing.</li>
  <li>Can cross-validate at each iteration.</li>
  <li>Tree pruning - unlike normal trees which stop adding branches, XGBoost will keep going deeper and then prune the tree</li>
</ul> 
XGBoost builds the first model and calculates the error for each observation in the dataset. It then builds a new model
to predict those errors (residuals). You are building a model based on the previous errors. This process gets repeately done.
A good video explaining this in further detail can be found <a href="https://www.youtube.com/watch?v=PxgVFp5a0E4">>here</a>.



<h2> Future Plans </h2>
 <ul>
  <li>Test out a deep learning model</li>
</ul> 