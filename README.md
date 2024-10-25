# Fraud Detection with Machine Learning

This repository contains a Jupyter Notebook that demonstrates the process of detecting fraudulent transactions using machine learning techniques. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization.

## Overview

Fraud detection is critical for financial institutions to prevent unauthorized transactions and minimize losses. This project aims to build a machine learning model that can classify transactions as fraudulent or legitimate based on various features.

## Dataset

The dataset used in this project (`Fraud.csv`) contains records of transactions, including details such as transaction type, amount, origin and destination accounts, and labels indicating whether a transaction is fraudulent.

### Columns:
- **step**: Time step of the transaction
- **type**: Type of transaction (e.g., PAYMENT, TRANSFER)
- **amount**: Transaction amount
- **nameOrig**: Account originating the transaction
- **oldbalanceOrg**: Balance before the transaction
- **newbalanceOrig**: Balance after the transaction
- **nameDest**: Account receiving the transaction
- **oldbalanceDest**: Balance of the destination account before the transaction
- **newbalanceDest**: Balance of the destination account after the transaction
- **isFraud**: Indicator if the transaction is fraudulent (1 = Fraud, 0 = Legit)
- **isFlaggedFraud**: Indicator if the transaction is flagged as fraudulent

## Requirements

To run the notebook, ensure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

- **Data Loading and Exploration**: The notebook starts by importing required libraries, loading the dataset, and performing an initial exploration to understand the data.
- **Data Preprocessing**: Steps to clean and preprocess the data, handle missing values, and perform feature engineering.
- **Model Training**: A RandomForestClassifier is used for training the model. Hyperparameter tuning is done using GridSearchCV.
- **Evaluation**: The model's performance is evaluated using various metrics, including accuracy, confusion matrix, classification report, and ROC-AUC score.
- **Visualization**: Visualizations are provided to help understand data distribution, feature importance, and model performance.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fraud-detection.git
    ```
2. Navigate to the project directory and open the Jupyter Notebook:
    ```bash
    cd fraud-detection
    jupyter notebook ShashankAccredianAssignment.ipynb
    ```
3. Run the cells sequentially to follow the data analysis and model-building process.

## Results

The notebook concludes with a section displaying the performance of the trained model, including metrics like precision, recall, and F1-score, along with visualizations for better insights into model predictions.

## Contributing

Contributions, issues, and feature requests are welcome! 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Feel free to modify or add any specific details to better fit your project!
