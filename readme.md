# Credit Card Default Prediction
## Demo Video link
```
https://youtu.be/zCjSYIT-I-k?si=6ZxtegmFZlZQhcJt
```


![alt text](image.png)

## Problem Statement:-
We can tackle this problem using machine learning. By analyzing a buyer's financial history, we can assess their creditworthiness. While we can't control companies' marketing tactics, we can proactively evaluate individuals' financial backgrounds to make informed decisions about lending or offering credit.

## Objective:-
The primary objective of this project is to leverage machine learning to predict whether a credit card user is likely to default on their payments. By evaluating past financial behaviors and patterns, we aim to provide credit decisions that are both responsible and sustainabl.

## Background Information:-
"Buy now, pay later" is a tempting offer in today's consumer-driven world. It allows us to satisfy our immediate desires without having the money upfront. However, this impulsive behavior often leads to mounting debt and financial distress, potentially pushing individuals into default or even fraudulent practices.



<h1><center><font size="6">Default of Credit Card Clients - Predictive Models</font></center></h1>



# Contents (Recommended)

- Introduction  
- Load packages 
- Read the data
- Check the data 
    - Glimpse the data
    - Check missing data
    - Check data imbalance
- Data exploration
- Predictive models
    - RandomForrestClassifier
- Conclusions




# Introduction


## Dataset

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from **April 2005** to **September 2005**. 

## Content

There are 25 variables:

* **ID**: ID of each client
* **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit
* **SEX**: Gender (1=male, 2=female)
* **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
* **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
* **AGE**: Age in years
* **PAY_0**: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
* **PAY_2**: Repayment status in August, 2005 (scale same as above)
* **PAY_3**: Repayment status in July, 2005 (scale same as above)
* **PAY_4**: Repayment status in June, 2005 (scale same as above)
* **PAY_5**: Repayment status in May, 2005 (scale same as above)
* **PAY_6**: Repayment status in April, 2005 (scale same as above)
* **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
* **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
* **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
* **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
* **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
* **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
* **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
* **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
* **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
* **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
* **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
* **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
* **default.payment.next.month**: Default payment (1=yes, 0=no)


## File Structure 
    .
    ├── app_exception           # Custom exception
    ├── application_logging     # custom logger
    ├── data_given              # Given Data
    ├── data                    # raw / processed/ transformed data
    ├── saved_models            # regression model
    ├── report                  # model parameter and pipeline reports.
    ├── notebook                # jupyter notebooks
    ├── src                     # Source files for project implementation
    ├── webapp                  # ml web application
    ├── dvc.yaml                # data version control pipeline.
    ├── app.py                  # Flask backend
    ├── param.yaml              # parameters
    ├── requirements.txt
    └── README.md



## Model information
Experiments:

         Model Name              R2 score 
      1. Linear Regression         77.92       
      2. Lasso Regression          82.03


## Installation
To run the code, first clone this repository and navigate to the project directory:
```
git clone https://github.com/Abhishek4209/Credit-Card-Default-Prediction.git
```
Create a virtual environment
```
conda create -p venv python==3.9 -y
conda activate venv/
```
To run this project, you will need Python packages present in the requirements file
```
pip install -r requirements.txt
```

Then, run the `app.py` file to start the Flask web application:
```
python app.py
```


### Setup
```pip install -e```

### Package building
``` python setup.py sdist bdist_wheel```

## Run the Project
- Clone the project
- pip install -r requirements.txt
- python app.py Enjoy the project in a local host

## Contributions
If you have any questions or suggestions regarding the project, please feel free to contact the project maintainer at `abhishekupadhyay9336@gmail.com`


## Conclusion

This project showcases how machine learning can be effectively used to predict credit card defaults. By analyzing historical financial data, we can make more informed and responsible credit decisions, ultimately contributing to a healthier financial ecosystem.