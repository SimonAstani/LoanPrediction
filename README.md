f# LoanPrediction
 Project Title: Loan Default Prediction using different machine learning algorithms
Project Title: Loan Approval
Prediction
Bowling Green State University
By
Olusegun Stephen Omotunde, Suman
Astani,Bishwas Kafle, Chandra Sekhar Ravuri
Department of Computer Science
CS 6010
Dr Shuteng Niu
Dec 4,20221. Introduction:
Loans are one of the core business services offered by banks to the public. The bank profits from
the interest on loans paid by customers. The bank or loan providers provide loans to an individual
after a deep investigation of verification and validation. Despite this, they don’t have any assurance
that applicants will repay their loans or not. Traditionally, most banks manually evaluate the risk
factor for providing loans to customers. Manual procedures are effective however, it is hard to
evaluate if there are more loan applications. It will take a long time to decide. Granting a loan to
the wrong applicant can result in bad debt in the long run for the bank or financial institution and
making the mistake of rejecting a loan applicant who met all the loan application standards can
result in the bank losing out on a potential client and source of income. when it comes to deciding
whether the applicant’s profile is relevant to be granted with loan or not. Banks must look after
many aspects.
We would like to use machine learning and data science knowledge to ease the work of banks and
other financial institutions and build models that can help predict whether loan applicants should
have their loans approved or not.
2. Problem Definition and Algorithms:
Loan prediction is a very common real-life problem that each bank or other financial institution
faces at least once in its lifetime. If done correctly, it can save a lot of resources at the end of the
banks or financial institutions. In this project, we will be predicting which of the loan applicants
should have their loans approved. It will help to automate the manual procedure of loan approval.
We will be using a publicly available dataset from Kaggle.
We will be conducting a series of steps to clean the data, analyze the dataset, build models using
different machine learning algorithms and evaluate the performance between those models.
3. Different Library used
1.
Pandas:
Pandas is an open-source python package that has numerous tools and features that can be used in
high performance data analysis and data manipulation. Different analysis and manipulating tasks
can be easily achieved with the help of this library. Data cleaning, merging, selecting. Similarly,
different file formats such as csv, json,sql can be easily imported.
2.
Dataprep:
Dataprep is a python library that allows data scientists to give a quick summary of pandas
dataframe with few lines of code. It is a Exploratory Data Analysis (EDA) tool that can help toexplore the data and understand its main feature. In the project, we used a simple create_report
function that generates an overall overview of the pandas dataframe object. It is the fastest and
easiest tool.
3.
Seaborn:
Seaborn is a data visualization library, used on python. This is based on Matplotlib. It is easy to
use and contains fascinating inbuilt themes compared to matplotlib. The syntax to use it is simple
and it is wiser to use it to work with pandas dataframe. Although matplotlib provides customization
power. We used seaborn in our project to visualize as it had the above-mentioned advantages.
4.
Sklearn:
Scikit learn is a machine learning library based on the Python programming language. Scikit learn
features various classification, regression, and clustering algorithms. Those include Support
Vector Machines, Random Forests, Gradient Boosting, K-means, etc. Using this library, we will
be building different models to train/test the data and compare accuracy between models to find
out the best performing one.
5.
Imbalanced-learn:
Imblearn is an open source, MIT-licensed library relying on scikit-learn (imported as sklearn) and
provides tools when dealing with classification with imbalanced classes.
4. Methodology
To predict the loan approval, we will be following standard steps for machine learning & data
science projects. We will follow the following steps.
4.1 Data Collection:
The data for the project has been collected from Kaggle repository. The data contains 13 different
features and 614 rows/records. We have 8 categorical variables: Gender (Male/Female),
Married(Yes/No), Education(Graduate/NotGraduate), No of dependents (0,1,2,3+), Self-
Employed(Yes/NO), Loan Status(Yes/No), Property Area(Rural/ Semi-urban/Urban), Credit
history. We have 4 numerical variables: Applicant Income, Co-applicant Income, Loan Amount,
Loan Amount term.The last variable/feature is Loan_ID.Fig: Categorical and Numerical Dataset
4.2 Data Cleaning:
We conducted a series of steps to clean the initial dataset. Initially, the dataset column names were
all listed in camelCase. We changed the column names into a standard Python style by converting
them to snake case style. Similarly, we also remove the features that are not useful for us. For
example, ‘loan__id’ is a unique identifier of the customer and is not useful to use for our project
because it does not provide any information or give any insight, but it would definitely be useful
on the bank's side. So, we dropped that column and updated the initial dataset.
#
Dropping
the
df.drop(columns=['loan__id'],inplace=True)
customer_id
column
We used dataprep library features to generate a DataPrep Report. It gives a detailed overview of
the distribution of the data statistics. Fig below highlights the overall overview of our data
generated from the library.Fig: Overall statistic of the dataset by DetaPrep library
From the above dataset, it can be clearly seen that a total of 149 data values are missing where
credit__history features got highest missing values and gender got least number of missing values.
Our approach to dealing with the missing values depended on the overall nature of the columns:
1. If it is a numeric column and tends to fall under normal distribution, then replace the
missing values with the mean of the column data i.e itself.
2. If it is a numeric column and right skewed/ left skewed fill the missing values with median
value or mode which is more robust than mean.
3. If it is a non-numeric or categorical column and right skewed/ left skewed, fill the
string/categorical columns with the mode. Each value will be replaced by the most frequent
value (mode).
4.3 Exploratory Data Analysis & Data Preprocessing:
Insights from Data Visualization: From the Exploratory Data Analysis, we could generate
insight from the data. How each of the features relates to the target variables.
Demographics Distribution (univariate categorical features)-- Gender Distribution: Males in the dataset are way more than females in the dataset i.e
81.76% to 18.24%. We have more male applicants
— Marital Status Distribution: Only around 34.69% of customers are not married. Majority of the
loan applicants are married.
Fig: Gender and marital status distribution in dataset
--- Education: more than 78% of customers are educated. Majority of the loan applicants are
educated.
–- Self Employed: Only about 13.36% of customers are self-employed. Majority of the loan
applicants are not self employed
Fig: Education and employment status distribution in dataset
— Dependents: Only 41.37% of customers have dependents. Majority of the loan applicants do
not have dependents.–- Property Area: Majority (37.95%) of the customers have property in the Semi-Urban Areas
followed by Urban areas (32.9%) with the remaining being in the rural areas. Maximum
properties are in Semi Urban areas
Fig: Dependents and property area distribution in dataset
Data Exploration (Bivariate Categorical features)
-- Gender - Males have more loans approved than females and they also have more loans
rejected than females. This makes sense because we have way more males in the dataset than
females.
-- Marital Status: We can see that married people had more chances of loan approval compared
to unmarried people.
Fig: Loan approval status based on gender and marital status
-- Dependents - Loan Applicants with no dependents have more loans approved than loan
applicants with dependents-- Education- Loan applicants who are graduates have their loans approved way more than
applicants who are not graduates
Fig: Loan approval status based on dependents and educational status
-- Self Employed - Loan applicants who are not self-employed have their loans approved way
more than applicants who are self-employed.
– Property Area: Applicants who owned property in the semi-urban area had higher chances of
loan approval compared to people in rural and urban areas.
Fig: Loan approval status based on employment status and property areaData Exploration (Target)
The distribution of the target shows that our data is imbalanced so we apply some oversampling
techniques like SMOTE (Oversampling methods duplicate) to create new synthetic examples in
the minority class so as to produce a more balanced data.
4.4 Model Building:
We performed Feature Scaling by using StandardScaler() to apply normalization on our
numerical features so as to standardize the input Xtrain to zero mean and unit variance before
fitting the estimator. We then performed feature encoding to handle our categorical variables by
applying One-Hot Encoding (OHE) using sklearn.
There exist various machine learning models for prediction. Each model has their own unique
features and modeling techniques. We have used the following models in our project and
performed comparisons between these models.We are dividing the total data set into two parts:
training data and testing data. Training data comprises 70 percent of the total dataset. The
remaining 30 percent of the dataset is used to test the model.
1. Linear Regression:
In the dataset, we have 13 features and one target variable. The target variable has Yes ( Y ) and
No ( N ) values. One of the models we are using is Linear Regression. For implementing the model
we have used Scikit learn. Since, linear regression only accepts numerical values, we are first
converting the target variable into linear values. Yes ( Y ) corresponds to 1 and No ( N )
corresponds to 0.2. Decision Tree:
Decision tree is a supervised machine learning model. Decision tree uses a series of possible
choices to produce an outcome. Starting from a single node, the decision tree branches out with
addition of each new feature. Each branch produces different possible outcomes.
3.Random Forest Classifier
Random forest classifier is a machine learning technique used for classification and regression. It
uses multiple decision trees and finally combines them to create a result. The resulting technique,
however, is different for classification and regression. The average result of each decision tree is
considered the result for regression. Whereas the result of the maximum number of decision trees
is the result for classification.
4. KNN Classifier
K-nearest neighbor is a supervised learning technique, which can be used for both classification
and regression. The KNN technique uses the K nearest point in the dataset.
5. MLP Classifier
MLP classifier implements Multilayer Perceptron algorithm. The algorithm uses three different
steps to train the model - forward pass, loss calculation and back propagation. The forward pass
just sees the input passed into a model and goes on calculating the weight and bias. At the end, the
loss is calculated from the predicted output and the output that is obtained. Finally, a back
propagation is done where required changes are made in the weights on each step traversing
backward.
4.5 Classification Evaluation Metrics of the models:
We have used 5 different classification evaluation metrics to analyze the result of our models. We
take a look at Accuracy, Precision, Recall, F1 score and ROC curve score of each model. Each of
them can be defined as:
Accuracy: Accuracy is the total performance of the model in all the classes. It can be described as
the general prediction. Accuracy is useful if all the classes are of equal importance.
Accuracy = ( True Positive + True Negative ) / Total
Accuracy is not a good metric for imbalanced datasets. Say we have an imbalanced dataset and a
badly performing model which always predicts for the majority class. This model would receive
a very good accuracy score as it predicted correctly for most observations, but this hides the trueperformance of the model which is objectively not good as it only predicts for one class. We
have dealt with our imbalanced data so now accuracy can be used as a good classification
evaluation metrics
Precision: Precision is an analyzing technique which determines how accurate are the predicted
positives in the result. In other words, it shows whether a predicted true positive is a false positive
or not. In our case, a true positive means that an applicant will have their loan approved, whereas
a false positive means that the model shows an individual whose loan should have been rejected
as approved, which is costly. Precision is intuitively the ability of the classifier not to label as
positive a sample that is negative. Higher Precision means we have less False Positives which is a
great thing in the context of our project. For our project Precision is one if not the most important
analysis because of the high cost associated with false positives.
Precision = True Positive/ ( True Positive + False Positive )
Recall: Recall is an analyzing technique which determines the model’s ability to predict true
positive outcomes. If the model predicts a loan applicant (actual true positive ) loan as not
approved ( false negative ), the bank is losing out on potential source of income by not loaning to
the right applicant and this is also costly. Higher Recall means we have less False Negatives.
Recall = True Positive / (True Positive + False Negative )
F1 score: F1 score is the balance between the Precision and Recall. The F1 score can be interpreted
as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and
worst score at 0.
The relative contribution of precision and recall to the F1 score are equal.
F1 score= 2 *(( Precision * Recall ) / ( Precision + Recall ))
ROC_Curve: The ROC curve of a good classifier is closer to the top left of the graph. Because in
a good classifier, we can achieve high TPR (true positive rate) at low FPR (false positive rate) at
optimal threshold.
The ROC curve helps us visualize how well our machine learning classifier is performing.
Although it works for only binary classification problems4.6 Analyzing Result:
The following figure shows the different classification evaluation metrics associated with different
models.
Fig1: Classification Evaluation metricsFig2: ROC_Curve
For our project, we used precision, roc curve, recall, accuracy and f1 score as our evaluation
metrics. Because we are working on a loan approval prediction problem, fp is more costly because
if our model is incorrectly predicting positive classes, then we are approving loan to applicants
that should have been rejected hence in the future such loans can become bad debts and the banks
and financial institutions want to avoid such situations. Hence precision is the most important
metric. Recall is also important because fn is also costly in our model because if the model is
incorrectly predicting negative classes, then we are rejecting loan applicants whose loan should
have been approved and hence the bank is losing out on potential customers and income. The
higher the recall the lower the fns hence this metrics is also important.
In our project, logistic regression produced the best results. Looking at Precision metric, the
logistics regression model produced 82% the second highest score and this indicates the second
least amount of time loan status was predicted wrongly. Logistic regression also produced the
second highest recall score of 91% and it produced the highest f1 score and accuracy as well.
Looking at fig2: the ROC curve, logistics regression also had the highest score which means it is
the best performing machine learning classifier.Confusion matrix of Logistics Regression
Fig:Confusion matrix
From the confusion matrix (1 as Yes loan approved and 0 as no loan rejected) of the logistic
regression above, we can see that there are 118 different instances of having a true positive
prediction, and 34 instances of true negatives. There were 9 instances of false negatives, and 24
instances of false positives.
These results can help banks and other financial institutions in their loan prediction process and
can help to reduce unnecessary cost or debts on the side of the banks & these institutions.5. Future work & Recommendations
[1] Better feature engineering (preprocessing/scaling) & Feature selection. Drop some features in
the dataset or even create new features in the dataset like total income by combining applicant
income & co applicant income then dropping these features.
[2] Collection of better/more data. Our dataset was relatively small, we believe with better datasets,
our result will improve
[3] Trying more Algorithms i.e more complex algorithms like xgboost, ensemble techniques like
bagging, bootstrap
[4] Hyperparameter tuning like gridsearch to identify the best parameters for each algorithm.
GridSearchCV is used to determine the parameters that gives the best predictive score for the
classifiers
[5] Performing model validation like cross validation.
6. References
[1] [PDF] Machine Learning in Banking Risk Management: A Literature Review | Semantic
Scholar
[2] Loan Prediction Project TermPaper
[3] Loan Prediction Problem Dataset | Kaggle
[4] https://github.com/omotuno/wecloud-project/blob/main/telecom_churn%20.ipynb
[5] Loan Prediction Dataset ML Project
| Kaggle
[6] Loan Approval Prediction Machine Learning by Vedansh Shrivastava, published on feb
4, 2022. Loan Approval Prediction Machine Learning - Analytics Vidhya
[7] Loan Prediction using Machine Learning Project Source Code
[8] Loan Prediction with Python. Preprocessing and classifying the… | by Andres Pinzon |
Medium[9] Ernest Owojori Loan prediction using selected machine learning, Nov 30, 2019. Loan
Prediction Using selected Machine Learning Algorithms | by Ernest Owojori | devcareers |
Medium
[10] Loan Approval Prediction using Machine Learning - GeeksforGeeks
[11] Loan Status with different models | Kaggle
[12]https://medium.com/@shetty.deeksha100/explore-your-dataset-within-seconds-thanks-
to-a-few-magical-codes-6fe178e029e0
[13] AUC-ROC Curve in Machine Learning Clearly Explained - Analytics Vidhya