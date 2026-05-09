car-mpg-prediction
Comparing Linear and Polynomial Regression to predict fuel efficiency (MPG) using the Auto-MPG dataset. Features EDA, correlation analysis, and model evaluation with Scikit-Learn.


Project Summary: Predicting Fuel Efficiency (Auto-MPG)

This project investigates the relationship between a car's Horsepower (engine power) and its MPG (fuel efficiency). The objective is to build a machine learning model that can accurately predict a car's mileage based on its technical specifications.

In this project, I compared two different modeling approaches:

Linear Regression (The Baseline):
What it is:A model that assumes a constant relationship. It draws a straight line through the data.
The Math: It uses OLS (Ordinary Least Squares) to minimize the distance between the line and the data points.
The Problem: It doesn't account for "diminishing returns" in fuel efficiency.


Quadratic Regression (The Improvement):
What it is: A type of Polynomial Regression (Degree 2). It allows the model to "bend" or curve.
Why it's better: Fuel efficiency doesn't drop at a constant rate. As engines get more powerful, the drop in MPG eventually levels off. The quadratic curve captures this "L-shaped" trend much better than a straight line.



Findings

Correlation: I found a strong negative correlation (-0.78) between Horsepower and MPG. As one goes up, the other goes down.
Performance: The Quadratic Model outperformed the Linear Model, achieving a higher R^2 score and a lower Mean Squared Error (MSE). This proves that the relationship between car power and efficiency is non-linear.
