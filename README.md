# Coursera Customer Churn Prediction
## Project Overview
* Implemented a neural network model to predict customer churn using tensorflow for a data science competition on Coursera
* Achieved a score for the evaluation metric (AUC) in the top 10% of submissions worldwide

## Intro
Subscription services are leveraged by companies across many industries, from fitness to video streaming to retail. One of the primary objectives of companies with subscription services is to decrease churn and ensure that users are retained as subscribers. In order to do this efficiently and systematically, many companies employ machine learning to predict which users are at the highest risk of churn, so that proper interventions can be effectively deployed to the right audience.

In this challenge, I predict churn for a group of subscribers on a video streaming service! 
Building a model that can predict which existing subscribers will continue their subscriptions for another month. The dataset is a sample of subscriptions initiated in 2021, all snapshotted at a particular date before the subscription was canceled. Subscription cancellation can happen for a multitude of reasons, including:
* The customer completes all content they were interested in, and no longer needs the subscription
* The customer finds themselves to be too busy and cancels their subscription until a later time
* The customer determines that the streaming service is not the best fit for them, so they cancel and look for something better suited

## EDA
**Dataset Info**

As explained above, the datasets are from video streaming platform and contain information about the customer, the customers streaming preferences, and their activity in the subscription thus far.

`train.csv` contains 70% of the overall sample (243,787 subscriptions to be exact) and importantly, will reveal whether or not the subscription was continued into the next month (the “ground truth”).

The `test.csv` dataset contains the exact same information about the remaining segment of the overall sample (104,480 subscriptions to be exact), but does not disclose `Churn` for each subscription.

Data descriptions for the initial dataset:
![alt text](https://github.com/mithilguru/Coursera-Customer-Churn-Prediction/blob/main/Visuals/Data-Descriptions.png)

**Data Validation**

Since predicting churn is a classfication task, the goal is building a baseline interpretable logistic regression model and moving forward from there.
Performed data validation first by checking for missing values, duplicate values, and mistyped data. No errors appeared in the dataset. Based on the given data descriptions, there are a number of categorical features, I'll need to refactor or encode these into numerical features to make modeling more efficient. 

**EDA**

![alt text](https://github.com/mithilguru/Coursera-Customer-Churn-Prediction/blob/main/Visuals/Churn-Variable.png)

Observing the values of `Churn` shows that the classes are imbalanced - there are far more returning customers than churned customers. This is important to note, I may need to balance the classes or apply cross-validation to improve model prediction accuracy.

![alt text](https://github.com/mithilguru/Coursera-Customer-Churn-Prediction/blob/main/Visuals/Categorical-Features.png)

Observing the rest of the cateogrical features in relation to `Churn` shows that the other features all have mostly balanced classes. There are no glaring differences between returning and churned customers for these fields.

![alt text](https://github.com/mithilguru/Coursera-Customer-Churn-Prediction/blob/main/Visuals/Numerical-Features.png)

Viewing numerical feature distributions for values of `Churn` reveals a few expected trends. Churned customers tend to have newer accounts, with less charges and viewing hours as a result. We can infer that newer customers are most likely to Churn.

![alt text](https://github.com/mithilguru/Coursera-Customer-Churn-Prediction/blob/main/Visuals/Correlation-Heatmap.png)

The range of the correlations shows that there are no new significant relationships between any of the variables except for `MonthlyCharges` and `TotalCharges` (expected). `AccountAge`, `MonthlyCharges`, and `SupportTicketsPerMonth` seem to be the most related to `Churn` out of all the predictors.

## Data Preprocessing / Feature Engineering

Cross-validated the dataset to prep for modeling with stratified k-fold sampling. This ensures each fold has the same ratio of the target class to properly evaluate the classifiers.

Performed mandatory compatiblity transformations next -
* Converted yes/no variables to 0/1 features
* One-hot encoded multi-class categorical features

## Modeling and Evaluation

Began modeling by fitting a baseline Logstic Regression model. It yields a modest .83 Accuracy and .55 AUC, meaning it performs reasonably well in terms of overall classification accuracy, but its ability to discriminate between positive and negative classes (churners and non-churners) on the validation set is not much better than random guessing. Suprisingly, test set AUC was .747, hinting at underfitting, as .747 AUC is in the top quartile of scores.

Possible reasons for lack of performance here include imperfectly optimized features and model hyperparameters, or potentially incorrect model type. I followed up by trying XGBoost classification next. XGBoost (Extreme Gradient Boosting) builds an ensemble of weak decision tree in a sequential manner, focusing on the instances that are difficult to classify correctly. The final prediction is a weighted sum of all of the tree predictions. This is likely a stronger choice to perform classification with due to the size and nature of the dataset.

The baseline XGBoost classifier suprisingly yielded weaker results than the first model (.825 Accuracy and .545 AUC) on the validation set, and test set AUC was also lower (.743). Optimizing the model via RandomizedSearchCV did not improve results past the baseline logistic regression classifier. 
Used feature importance next to determine the most relevant predictors, XGBoost still lagged in performance, while the Logistic Regression AUC improved to .749.

## Final Model and Results

Since the previous techniques (Logistic Regression, XGBoost Classification) were relatively similar in performance with model and data enhancements only slightly boosting scores, deep learning was my final choice for predicting churn. NNs can recognize hidden patterns in the data and generate predictions accordingly. Re-implemented feature selection, scaling, and cross-validation for the final submission, along with neural architecture search to optimize the model structure.

The result is a final validation accuracy of .835 and a **.75 AUC**, netting a top 10% submission.
