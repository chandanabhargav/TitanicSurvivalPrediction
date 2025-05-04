import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, render_template_string
import io
import base64
import os

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Load dataset
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

    # Handle missing values
    df['Age'] = df['Age'].median()
    df = df.drop('Cabin', axis=1)

    # Prepare features and target
    X = df.drop(columns=['PassengerId', 'Survived'], axis=1)
    y = df['Survived']
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Plotting
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_plot = numerical_cols[:min(len(numerical_cols), 6)]
    images = []

    # Helper function to convert plots to base64
    def plot_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        return img_base64

    # Histogram
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(2, 3, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(col)
    plt.suptitle("Histograms for Selected Columns", fontsize=16)
    plt.tight_layout()
    images.append(plot_to_base64())

    # Boxplot
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.suptitle("Box Plots for Selected Columns", fontsize=16)
    plt.tight_layout()
    images.append(plot_to_base64())

    # Countplot
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(2, 3, i + 1)
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.xticks(rotation=45, ha='right')
        plt.title(col)
    plt.suptitle("Count Plots for Selected Columns", fontsize=16)
    plt.tight_layout()
    images.append(plot_to_base64())

    # Heatmap
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm')
    plt.title("Heatmap for Selected Columns", fontsize=16)
    images.append(plot_to_base64())

    # Render HTML with embedded images
    html = '''
    <html>
    <head><title>Titanic Model Analysis</title></head>
    <body style="font-family: Arial, sans-serif;">
        <h2>Model Accuracy: {{ accuracy }}</h2>
        {% for img in images %}
            <div style="margin-bottom: 40px;">
                <img src="data:image/png;base64,{{ img }}" style="max-width: 100%;" />
            </div>
            <hr>
        {% endfor %}
    </body>
    </html>
    '''
    return render_template_string(html, accuracy=round(accuracy, 4), images=images)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
