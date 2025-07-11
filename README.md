# Lab 2: KNN vs RNN Classifier on the Wine Dataset

**Author:** Jacob Jeffers  
**Course:** MSCS 634 – Data Mining  

## Overview
In this lab, I explored how K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) perform on the Wine dataset from `sklearn`. The dataset contains chemical features for different types of wine. I wanted to understand how changing the number of neighbors (`k`) or the radius affected classification accuracy.

## What I Did
- Loaded and explored the Wine dataset
- Split the data into training and testing sets
- Trained KNN models using different `k` values: 1, 5, 11, 15, and 21
- Trained RNN models using radius values between 350 and 600
- Measured and recorded accuracy for each setting
- Created plots to visualize the trends in performance

## What I Noticed
- KNN gave steady and reliable results. The best accuracy showed up with `k = 5` and `k = 11`
- RNN was more unpredictable. When the radius was too small or too large, it returned outlier labels for some test points, which I had to filter out
- KNN was simpler to tune and more consistent. RNN required more guessing and adjustment

## Challenges I Ran Into
The main challenge was with RNN. If no neighbors were found within the specified radius, the model labeled the prediction as an outlier (`-1`). These had to be removed before calculating accuracy, or else they would throw things off. This step added a bit of extra work compared to KNN, which ran more smoothly.

## Files Included
- `lab2.ipynb`: The Jupyter Notebook with all code, output, and plots
- `README.md`: This summary of the project
