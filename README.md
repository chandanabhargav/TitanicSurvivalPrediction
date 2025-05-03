
# Titanic Model Analysis Flask App

This Flask web application performs a basic machine learning analysis on the Titanic dataset using logistic regression and displays visualizations to understand the data.

## Features

- Loads Titanic dataset from a public URL
- Cleans missing values and prepares features
- Trains a logistic regression model to predict survival
- Calculates and displays model accuracy
- Generates the following plots:
  - Histograms for selected numerical columns
  - Box plots for selected numerical columns
  - Count plots for selected numerical columns
  - Heatmap of correlations

## Setup and Usage

1. Install dependencies:

```
pip install flask pandas matplotlib seaborn scikit-learn
```

2. Run the Flask app:

```
python predict.py
```

3. Open your browser and go to:

```
http://localhost:5000/predict
```

## Output

The app will display model accuracy and four different types of plots based on the dataset.

---

Dataset Source: [Titanic Dataset on GitHub](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)
