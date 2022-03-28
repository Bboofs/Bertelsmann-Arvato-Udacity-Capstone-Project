![Python](https://img.shields.io/badge/Python-3.7-brightgreen)
![flask](https://img.shields.io/badge/Flask-2.0.3-green)
![pandas==1.3.5](https://img.shields.io/badge/Pandas-1.3.5-yellowgreen)
![cmake==3.22.3](https://img.shields.io/badge/cmake-3.22.3-yellow)
![numpy==1.21.5](https://img.shields.io/badge/Numpy-1.21.5-orange)
![jupyter==1.0.0](https://img.shields.io/badge/Jupyter-1.0.0-red)
![xgboost==1.5.2](https://img.shields.io/badge/XGBoost-1.5.2-blue)
![openpyxl==3.0.9](https://img.shields.io/badge/openpyxl-3.0.9-lightgrey)
![seaborn==0.11.2](https://img.shields.io/badge/Seaborn-0.11.2-blueviolet)
![flask_wtf==1.0.0](https://img.shields.io/badge/Flask%20WTF-1.0.0-brightgreen)
![notebook==6.4.10](https://img.shields.io/badge/Notebook-6.4.10-green)
![matplotlib==3.5.1](https://img.shields.io/badge/matplotlib-3.5.1-yellowgreen)
![scikit-learn==1.0.2](https://img.shields.io/badge/scikit%0Alearn-1.0.2-yellow)
![flask_bootstrap==3.3.7.1](https://img.shields.io/badge/flask_bootstrap-3.3.7.1-orange)

# Udacity Data Science Nanodegree

## Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services

All the code is in a public repository [here](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project)

## Table of Contents
1. [Installation](#installation)
1. [Running the Web App](#run_app)
1. [Project Definition](#definition)
1. [File Descriptions](#file_desc)
1. [Results](#results)
1. [Project Analysis](#analysis)
1. [Project Conclusion](#conclusion)

## Installation <a name="installation"></a>

By installing the packages in the `requirements.txt`, the dependencies will be installed.

The main packages and their versions are listed below as exist in requirements file, and are cumulative of the packages required for the web and the data analysis.

- flask==2.0.3
- pandas==1.3.5
- cmake==3.22.3
- numpy==1.21.5
- jupyter==1.0.0
- xgboost==1.5.2
- openpyxl==3.0.9
- seaborn==0.11.2
- flask_wtf==1.0.0
- notebook==6.4.10
- matplotlib==3.5.1
- scikit-learn==1.0.2
- flask_bootstrap==3.3.7.1

## Running the Web App <a name="run_app"></a>

We were given the option of submitting analysis together with either a blog post or a web application. If a web 
application is submitted, then the `README.md` file must have:
- Project Definition
- Project Analysis
- Project Conclusion
 I've opted for the web application and the 3 items above are covered in this file at a later section.
 
 The web app is a very basic Flask application that has only two routes and a form using 
 [Flask-WTF](https://flask-wtf.readthedocs.io/en/stable/) and 
 [Flask-Bootstrap](https://pythonhosted.org/Flask-Bootstrap/index.html).
 
 When you run it for the first time, it loads the form which is dynamically generated and prepopulated with initial 
 values due to the large number of fields in the form (349).
 
 All the fields in the form correspond to the columns in the cleaned data with only allowed possible values as 
 selectable options. You are free to select your own options and submit to see different results.
 
 >**Note:**
 I was expecting that from the results page, clicking the lick to home page would automatically load the form afresh but 
 it comes wih previous data so on the second time you'll have to modify the values yourself or stop the application and 
 start it afresh.  

Run the app by going into the project folder and executing the command below:

`python ./web_app/customers.py`

Then open your web browser and type this in the address bar:

`localhost:4321/`

>**Note** if the Flask web server was already running before this, you'll need to stop it by pressing Control-C in 
terminal before you run it again.

Below is the an example of first page:

![First Page](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/web_app_home.JPG)

On submission, you get second page with the predicion as below:

![Second Page](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/web_app_pred.JPG)

## Project Definition <a name="definition"></a>

This project is part of the requirements for the completion of Udacity's Data Science nanodegree. In the course, it is 
called **The Capstone Project** and involves a requirement to analyse demographics data for customers of a mail-order 
sales company in Germany and proceed to compare it against demographics information for the general population.

The datasets used were provided by Bertelsmann Arvato Analytics, and the project represents a real-life data science task.
This data has been strictly used for this project only and will be removed from my laptop after completion of the project
in accordance with Bertelsmann Arvato terms and conditions.

The project has 3 sections as guided in the template notebook provided at the start of the project

The first section is the **Customer Segmentation Section** which requires to use unsupervised learning techniques to 
analyse features of established customers and the general population in order to create customer segments with the aim 
of targeting them in marketing campaigns.

The second section is the **Supervised Learning Model Section** in which a supervised machine learning model that 
predicts whether or not each individual will respond to the campaign has been built.

The third section is the **Submission Sections** where the most promising model is fine-tuned and is then used to make 
predictions on the campaign data.

## File Descriptions <a name="file_desc"></a>

The data provided is not publicly available according to Bertelsmann Arvato T&Cs.

![Inside Project Folder](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/inside_project_folder.JPG)

The code is contained in following files:

  - **Arvato Project Workbook.ipynb** - This is the main and the only Jupyter Notebook and it contains
    - the EDA of the general population and customers data where the data preprocessing steps are also identified.
    - the unsupervised learning model (Customer Segmentation) section
    - the supervised learning model section
  - **data_preprocessing.py** - Contains all the functions that are used in the notebook to preprocess the data and 
  display some of the graphs
  - **web_app** - contains the files necessary for the web to function
  - **web_app/customers.py** - the file that generates the form, processes the input data, loads the model and 
  generates the prediction value
  - **web_app/templates/** - folder with the html templates
  - **data/** - folder with data sources and images used in the project
    - Udacity_AZDIAS_052018.csv: Demographics data for the general population
    - Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mailorder company
    - Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign
    - Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign
    - Additional pickle: files for saving data processing states and models


## Project Analysis <a name="analysis"></a>

### EDA Section

A large number of the fields did not have their descriptions in the attributes file. This was made worse by abbreviations 
in the column names and use of a language other English to name the columns.

Quite a big number of the columns had nulls. Before any cleaning was done, a graph of top 15 columns with nulls was 
plotted, with some columns nearly having everything as null.

![null_percentages](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/null_percentages.png)

In addition to that, even the description had issues and had to under go cleaning to make sure only valid values were 
in the file. invalid values were where there are multiple options instead of just one possible entry and nulls.
After this clean up on the attributes file, even more columns got more nulls.

![more_null_percentages](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/more_null_percentages.png)

However, the missing values seemed to have consistency in bothe the customers profile data and the general population data.
Also, all the data had very low cardinality.

![missing_consistency](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/missing_consistency.png)

Other elements of the EDA included encoding the categorical columns, imputing the null values and scaling the data.

### Customer Segmentation Section

Owing to to the fact that some of the features did not have their meanings described in the attributes file, it was 
quite hard to describe some of the clusters generated. Where possible, it would be better to drop such features or go back to the 
original sources where the data was collected to seek clarification, or to the people with domain knowledge. 

Despite the data challenges, I was able to generate the following groups of potential customers:

1. The population with good financial standing:
    - Such individuals have high-end cars with high income
1. The population that live in good neighbourhoods and minicipalities
    - They also tend to have multiple cars
1. The population with relatively young individuals
    - They tend to have interest in environmental sustainability but appear to be disappointed with their current social
     standing and interested in low interest rates financial services.

These groups represent the part of the population which if the company decided to launch any marketing campaigns, they 
would be good targets.

Additional groups within the population that have been identified shows under-representation in the customer base. These 
include:
1. The population who tend to be young and have cars with medium incomes.
1. The population tha save money with low interest in financial services.
1. The population hat tend to e younger with low income.

For the company to reach a larger part of the population, it might need to design some financial services that align 
with with the needs of the above groups. However, these profile of people do not spend money often and the company would 
need to look into profitability before launching such services

### Supervised Learning Model Section

For starters, the dayta was heavily imbalanced between those who responded and those who did not respond:

![class_imbalance](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/class_imbalance.png)

This meant that even if the model predicted zeroes only for all the data points we would still end up with having a high 
which would not be correct in this case.

As a result, to address this imbalance while evaluating the model requires choosing a metric which will consider the 
class imbalance. Such metrics include:
- Precision
- Recall
- Area Under Receiver Operating Curve (AUROC).

After training, using the AdaBoost, the performance of ROC AUC was 0.7424 while using the XGBClassifier, the performance 
of ROC AUC score was 0.7469.

However, AdaBoost seemed to give importance to only one feature while XGBoost seemed to have a more distributed feature 
importance.

![ada_boost_feature_importance](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/ada_boost_feature_importance.png)

![xgboost_feature_importance.png](https://github.com/Bboofs/Bertelsmann-Arvato-Udacity-Capstone-Project/blob/main/data/images/xgboost_feature_importance.png)


## Project Conclusion <a name="conclusion"></a>
Being new to data science, this project was very complicated. The features were very many and difficult to track. This 
was made worse by the slow system that timed out a number of times and I had to restart running the whole thing a fresh 
each time this happened, making the progress even slower.

However, it offered key insights into real world data science projects and prepared me to be ready to take on  the next 
assignment.