---

# Credit Card Fraud Detection

This project aims to predict whether a credit card holder will default on their payment based on their financial history and other factors. The model uses machine learning techniques and big data tools to identify potential fraud or defaults, helping financial institutions make informed decisions.

## Problem Statement
Credit card fraud and defaults are a significant concern for financial institutions. By predicting potential defaulters, institutions can reduce the risk associated with credit issuance. In this project, we use a dataset containing information about credit card clients to build a model that can predict whether an individual will default on their payment.

## Objective
The primary goal of this project is to apply machine learning models, alongside big data technologies like **Apache Spark**, to predict credit card default payments. By leveraging a variety of demographic, credit history, and financial data, we aim to build an accurate predictive model that helps identify clients who may default on payments.

## Dataset Overview
This dataset contains information about credit card clients in Taiwan and includes several demographic features, credit data, and payment history. The features are used to predict whether a person will default on their payment in the subsequent month.

### Features:
1. **ID**: Unique identifier for each client
2. **LIMIT_BAL**: Credit limit given to the client
3. **SEX**: Gender of the client (1 = Male, 2 = Female)
4. **EDUCATION**: Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)
5. **MARRIAGE**: Marital status (1 = Married, 2 = Single, 3 = Others)
6. **AGE**: Age of the client
7. **PAY_0** to **PAY_6**: Repayment status for the past 6 months
8. **BILL_AMT1** to **BILL_AMT6**: Bill statements for the past 6 months
9. **PAY_AMT1** to **PAY_AMT6**: Payment amounts for the past 6 months
10. **default.payment.next.month**: Default status (1 = Yes, 0 = No)

### Task
We need to classify each client as either a defaulter (1) or not (0) based on the provided features. This is a binary classification problem.

## Technologies Used
- **Apache Spark**: Used for distributed data processing to handle large datasets efficiently.
- **Python**: Primary programming language for data analysis and model implementation.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing machine learning models and evaluation metrics.
- **Flask**: Used to deploy the machine learning model in a web application.
- **Matplotlib/Seaborn**: For data visualization and insights generation.
- **NumPy**: For numerical operations.
- **DVC (Data Version Control)**: To manage data versioning and reproducibility of experiments.

## Data Analysis and Preprocessing
We performed various preprocessing steps to ensure the dataset is clean and ready for modeling:
- **Handling Missing Values**: Ensured no missing values in the dataset.
- **Normalization**: Scaled numerical features to bring them onto the same scale.
- **Feature Engineering**: Created new features where necessary and dropped irrelevant ones.
- **Data Splitting**: Split the data into training and testing datasets to evaluate the model's performance.

## Model Selection and Performance
### Model: Random Forest Classifier
We implemented multiple machine learning models, but the **Random Forest Classifier** achieved the highest performance.

- **Accuracy**: 97%
- **Precision**: 95%
- **Recall**: 96%
- **F1-Score**: 95%

This model performed exceptionally well and provided robust predictions, making it the ideal choice for this task.

## File Structure

```
.
├── app_exception            # Custom exceptions
├── application_logging      # Custom logging utility
├── data_given               # Given raw data
├── data                    # Processed and cleaned data
├── saved_models            # Trained models
├── report                  # Model evaluation reports
├── notebook                # Jupyter notebooks for data analysis and model training
├── src                     # Source code for the project
├── webapp                  # Flask web application for model deployment
├── dvc.yaml                # Data version control pipeline
├── app.py                  # Flask backend for the web app
├── param.yaml              # Configuration parameters
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation
To set up the project, follow these steps:

1. Clone this repository:
   ```
   git clone https://github.com/Lavishgangwani/Credit-Card-Fraud-Detection.git
   ```

2. Create a virtual environment:
   ```
   conda create -p venv python==3.9 -y
   conda activate venv/
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the web application:
   ```
   python app.py
   ```

## Model Evaluation
The **Random Forest Classifier** performed the best in terms of accuracy, with a **97% accuracy** rate on the test set. It outperformed other models such as Logistic Regression and Support Vector Machines (SVM) in terms of precision and recall, making it the most reliable choice for detecting credit card defaulters.

### Performance Metrics:
- **Accuracy**: 97%
- **Precision**: 95%
- **Recall**: 96%
- **F1-Score**: 95%

These metrics indicate that the Random Forest Classifier is highly effective at distinguishing between defaulters and non-defaulters.

## Apache Spark Integration
Apache Spark was integrated into this project to handle the large volume of data efficiently. Using Spark's distributed computing capabilities, we were able to preprocess and train models on large datasets without sacrificing performance. It enabled faster training times and better scalability for future expansions.

### Key Benefits:
- **Distributed Data Processing**: Spark's distributed nature allowed us to process the dataset efficiently.
- **Faster Computations**: Spark significantly reduced the time required to preprocess and train models compared to a traditional single-machine approach.

## Conclusion
This project demonstrates how machine learning techniques, in conjunction with big data tools like **Apache Spark**, can be applied to predict credit card default payments. By utilizing demographic data, financial history, and payment records, we built a robust model that can help financial institutions make informed decisions about credit issuance.

## How to Contribute
If you have any suggestions or improvements, feel free to contribute to this project. You can fork the repository, make changes, and submit a pull request.

For any questions or inquiries, you can reach the project maintainer at `lavishgangwani22@gmail.com`.

---