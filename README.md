# House Price Prediction

## Overview
This project aims to predict house prices using historical housing sale data. It includes preprocessing steps such as data cleaning and feature engineering to enhance the dataset's quality and relevance. The processed data is then used to train an XGB (eXtreme Gradient Boosting) machine learning algorithm, a powerful tool for regression tasks like house price prediction.

## Exploratory Data Analysis (EDA)
Key insights were uncovered during the EDA stage, revealing patterns and trends in the housing market:

![Sale Price Distribution](sale-price-distribution.png)
![Feature Importance (Small)](feature-importance-small.png)
![Feature Importance](feature-importance.png)

## What is XGBoost and How Does It Work?
XGBoost stands for eXtreme Gradient Boosted trees. It is renowned in the machine learning community, particularly for its performance in Kaggle competitions. XGBoost is an ensemble method that creates multiple model iterations, each focusing on correcting the residuals (errors) of the previous one. This process utilizes a gradient descent algorithm to minimize loss. Key features include:
- Regularization (L1 and L2) to prevent overfitting.
- Parallel processing for efficient computation.
- Cross-validation capabilities at each iteration.
- Advanced tree pruning techniques.

For an in-depth understanding of XGBoost, watch this explanatory [video](https://www.youtube.com/watch?v=PxgVFp5a0E4). Also, see this [Stats StackExchange post](https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest) for a comparison with Random Forest and traditional Gradient Boosting.

### How is XGBoost Different from Traditional Boosting?
- XGBoost leverages second-order gradient information to find the optimal direction for reducing errors.
- It incorporates both L1 and L2 regularization, enhancing model performance.
- Parallelized tree construction boosts computational speed.

## Future Plans
- Experimentation with deep learning models, possibly using neural networks with frameworks like TensorFlow or PyTorch, to compare their efficacy against traditional machine learning approaches in house price prediction.

