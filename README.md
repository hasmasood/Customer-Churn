# Customer-Churn
A dummy classification project to predict probability of customer churning.

## Technology stack
[Windows Subsytem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)

[Amazon Web Services (AWS)](https://aws.amazon.com/)

[Visual Studio Code](https://code.visualstudio.com/)

[Docker](https://www.docker.com/)

## Working environment
Create working environment aws_env using environment.yml file
```
conda env create -f environment.yml
```

## Deployment environment
We create a separate environment to deploy the code, working on Python 3.7.0.
```
conda create -n depl_env python==3.7.0
```
Install the required libraries (listed in requirements.txt) in the deployment environment.
```
conda activate depl_env
pip install -r requirements.txt
```

## Contents
[Step1-EDA.ipynb](https://github.com/hasmasood/Customer-Churn/blob/main/Step1-EDA.ipynb)
A simple exploratory data analysis by sourcing raw data from one S3 bucket and exporting processed data to another bucket. An important tool facilitating the understanding, cleaning and wrangling of data is ```pandas_profiling```. Unlike regression project, ``` featurewiz ``` isn't used here.

[Step2-EDA-detailed.ipynb](https://github.com/hasmasood/Customer-Churn/blob/main/Step2-EDA-detailed.ipynb)
An extension of exploratory data analysis. Dictionary vectorizer was used to encode categorical features, unlike one-hot encoder of pandas in the other regression project. Indepth information on mutual interaction of variables is also included.

[Step3-MLDev.ipynb](https://github.com/hasmasood/Customer-Churn/blob/main/Step3-MLDev.ipynb)
This describes the process of feature engineering, machine learning model development, assessment and visualization of predictive performance. A number of models are evaluated and hyperparameters tuned via grid search pipelines. The best model selected is later trained from scratch. Models, scaler function and dictionary vectorizer are exported using ```pickle``` in a single stacked file.

 [Step4-Deploy.ipynb](https://github.com/hasmasood/Customer-Churn/blob/main/Step4-Deploy.ipynb)
 This deploys the model both locally as well as web service and make predictions for a sample data. 