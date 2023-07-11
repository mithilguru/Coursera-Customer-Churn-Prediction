# Coursera Customer Churn Prediction
## Project Overview
* Implemented a neural network model to predict customer churn using tensorflow for a data science competition on Coursera
* Achieved a score for the evaluation metric (AUC) in the top 10% of submissions worldwide

## Intro
Subscription services are leveraged by companies across many industries, from fitness to video streaming to retail. One of the primary objectives of companies with subscription services is to decrease churn and ensure that users are retained as subscribers. In order to do this efficiently and systematically, many companies employ machine learning to predict which users are at the highest risk of churn, so that proper interventions can be effectively deployed to the right audience.

In this challenge, I'll be predicting churn for a group of subscribers on a video streaming service! 
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
![alt text](https://github.com/mithilguru/Fitbit-Tracker-Insights/blob/main/Visuals/Usage-DoW.png?raw=true "Total Logs by Day of Week")

## Data Preprocessing / Feature Engineering

## Modeling

## Results

