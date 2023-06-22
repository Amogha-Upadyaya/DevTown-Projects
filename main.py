# Import Pandas and NumPy

# importing numpy be used as an efficient multi-dimensional container of generic data.
import numpy as np

#importing pandas it allows you to perform data manipulation create, manipulate and wrangle the data in python.
import pandas as pd


#   ---------- # Visualizations---------

# Import Libraries for plotting

#Matplotlib is a Python 2D plotting library
import matplotlib.pyplot as plt
plt.show()

# Seaborn is high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Read the data of credit card fraud detection
# Here file is CSV i.e Comma Seperated Values
#df is a variable

df = pd.read_csv("creditcard.csv")

df.head()

# Checking the first 5 entries of dataset

# # To know the information of the dataset

df.info()

print("There are {} rows and {} columns are present in the Data Set".format(df.shape[0],df.shape[1]))

#printing the rows and columns

df.describe()
##describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values.

df.isnull().sum()

##isnull(). sum() returns the number of missing values in the data set.

#visualizing the null values for each attribute


import missingno as msno

#msgo ---->  It's also the name of a Python library for the exploratory visualization of missing data.

msno.bar(df)
#ploting the bar graph

plt.show()
#Show the image

df.columns
# To check what are the columns are present in the dataset

#creating a variable called lst
# creting a list (Name of variable is lst)
lst=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class']

for i in lst[1:]:  # iterating all the rows
    df[i].hist(bins=50, figsize=(10, 6))

    # Width of each bin is = (max value of data – min value of data) / total number of bins
    # hist means histogram, here we using with the help of matplotlib , it gives some bins to understand bars

    plt.yscale('log')
    # the type of conversion of the scale, to convert y-axes to logarithmic scale we pass the “log” keyword or the matplotlib. scale
    # LogScale class to the yscale method
    plt.title(i)

    plt.show()
    # Show the image

df.drop(columns=['Time','Class'])

# dropping time,id columns

# plotting correlation plot

plt.figure(figsize=(20,15))
#plotting the figure size based on width and height

sns.heatmap(df.corr(),cmap='PiYG',annot=True,linewidths=1,fmt='0.2f')

# data: 2D dataset that can be coerced into an ndarray.
# cmap: The mapping from data values to color space.
# annot: If True, write the data value in each cell.
# fmt: String formatting code to use when adding annotations.
# linewidths: Width of the lines that will divide each cell.

#  create dataset

X = df.iloc[:,:-1] #independent Variable

y = df.iloc[:,-1] #dependent Variable

y

sns.countplot(x='Class', data = df)
#is used to Show the counts of observations in each categorical bin using bar


from collections import Counter
#Counter is a container which stores the count of elements in a dictionary format where element is the key and its value corrosponds to it's count.

counter = Counter( df [ 'Class' ])
#passing 'Class' feature in the Counter , it tells no. of 1s and 0s present in the dataset

print(counter)
#print the counter variable

# 1 ------->    Fraud
# 0 ------->   Not Fraud

np.random.seed(1001)
# np.random.seed   it can generate same random numbers on multiple executions of the code on the same machine

#importing train_test_split
from sklearn.model_selection import train_test_split

## split into train tests sets
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2,stratify=y
                                              )

# Check the shape of each
x_train.shape,x_test.shape,y_train.shape,y_test.shape

#Use scikit-learn’s NearestNeighbors:
from sklearn.neighbors import KNeighborsClassifier

# Build a model
# Find each observation's five  nearest neighbors
# based on minkowski distance (including itself)

## Train a KNN classifier with 5 neighbors
model = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)



# minkowski ----> The Minkowski distance is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance

model.fit(x_train,y_train)

#Train the model

#predict the model on test
y_pred=model.predict(x_test)

#predict the models and probabilities
y_pred_proba=model.predict_proba(x_test)[:,1]

# import the libraries for stats metrices
import numpy as np

# importing the Confusion matrix metrics and classification reports
from sklearn.metrics import confusion_matrix, classification_report

# Importing cohen_kappa_score and roc_auc_score metrices
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.metrics import roc_curve, auc

# importing visualizing library
import matplotlib.pyplot as plt
import seaborn as sns

# logloss to check is there loss or difference
from sklearn.metrics import log_loss


# Creating a Function name called Classification Metric
def classification_metric(y_test, y_pred, y_prob, label, n=1, verbose=False):
    """
    Note: only for binary classification
    confusionmatrix(y_true,y_pred,labels=['No','Yes'])
    """
    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    row_sum = cm.sum(axis=0)
    cm = np.append(cm, row_sum.reshape(1, -1), axis=0)
    col_sum = cm.sum(axis=1)
    cm = np.append(cm, col_sum.reshape(-1, 1), axis=1)

    labels = label + ['Total']

    plt.figure(figsize=(10, 6))
    # plotting a fig size as 10 width and 6 height

    sns.heatmap(cm, annot=True, cmap='summer', fmt='0.2f', xticklabels=labels,
                yticklabels=labels, linewidths=3, cbar=None, )
    # create a heapmap using seaborn libarary and used various parametere

    plt.xlabel('Predicted Values')
    # ploting the values on x- axis as Predicted values

    plt.ylabel('Actual Values')
    # ploting the values on y- axis as actual values

    plt.title('Confusion Matrix')
    # Mentioning the title of the figure

    plt.show()
    # show the image

    print('*' * 30 + 'Classifcation Report' + '*' * 30 + '\n\n')
    # showing * are to put a  line to style

    # created classification report
    cr = classification_report(y_test, y_pred)

    # print the classifiaction report
    print(cr)

    print('\n' + '*' * 36 + 'Kappa Score' + '*' * 36 + '\n\n')

    # Kappa score
    kappa = cohen_kappa_score(y_test, y_pred)  # Kappa Score
    print('Kappa Score =', kappa)

    print('\n' + '*' * 30 + 'Area Under Curve Score' + '*' * 30 + '\n\n')
    # Kappa score
    roc_a = roc_auc_score(y_test, y_pred)  # Kappa Score
    print('AUC Score =', roc_a)

    # ROC

    plt.figure(figsize=(8, 5))
    # plot the figuare based on width and height sizes

    fpr, tpr, thresh = roc_curve(y_test, y_prob)
    # fpr false positive rate
    # tpr true positive rate

    plt.plot(fpr, tpr, 'r')
    print('Number of probabilities to build ROC =', len(fpr))
    if verbose == True:
        for i in range(len(thresh)):
            if i % n == 0:
                plt.text(fpr[i], tpr[i], '%0.2f' % thresh[i])
                plt.plot(fpr[i], tpr[i], 'v')

    plt.xlabel('False Positive Rate')
    # fpr on x -axis

    plt.ylabel('True Positive Rate')
    # tpr on y axis

    plt.title('Receiver Operating Characterstic')
    # mentioning the title of the figuare

    plt.legend(['AUC = {}'.format(roc_a)])
    # assign the legend to the figuare

    plt.plot([0, 1], [0, 1], 'b--', linewidth=2.0)
    # mentioning then line width as 2.0

    plt.grid()
    # show the grid lines to the image

    plt.show()
    # display the image


# A point beyond which there is a change in the manner a program executes
class threshold():
    '''
    Setting up the threshold points
    '''

    def __init__(self):
        self.th = 0.5

    def predict_threshold(self, y):
        if y >= self.th:
            return 1
        else:
            return 0

#Calling the Classification_metric function. It will displays all the metrices which are we created earlier

classification_metric(y_test,y_pred,y_pred_proba,['no','yes'],n=1,verbose=True)

# VERBOSE : This flag allows you to write regular expressions that look nicer and are more readable by allowing you to visually separate logical sections of the pattern and add comments.

#importing imblearn library
import imblearn

from imblearn.over_sampling import SMOTE

#efine dataset, mentioning neighbours is 5 and fit the model
x_resample,y_resample=SMOTE(k_neighbors=5).fit_resample(df,y)

x_resample

# Saving arrays name called credit_card_oversample
np.savez('credit_card_oversample.npz',x_resample,y_resample)

pd.Series(y_resample).value_counts()
##value_counts() function returns object containing counts of unique values

#load data what we balanced the data
data_over=np.load('credit_card_oversample.npz')
#read it
data_over.files

x_over=data_over['arr_0']

#independent variable

y_over=data_over['arr_1']   #dependent variable

pd.Series(y_over).value_counts()
#Create it
#value_counts() function returns object containing counts of unique values

#import the sklean and train_test_split
from sklearn.model_selection import train_test_split

#Split the data into train and test , mention test_size
x_train,x_test,y_train,y_test=train_test_split(x_over,y_over,test_size=0.2)

# Check the shape of the each data
x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.neighbors import KNeighborsClassifier
# Import the algorithm, called KNN

#Build the model and mention n_neighbours, and metric
model_over = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

model_over.fit(x_train,y_train)
#train the model

y_pred=model_over.predict(x_test)
#predict the model

y_pred_prob=model_over.predict_proba(x_test)[:,1]
#predict and find probabilities

classification_metric(y_test,y_pred,y_pred_prob,['no','yes'],n=1,verbose=True)

#import the train_test_split
from sklearn.model_selection import train_test_split

#split the data into train , test and mention test size
x_train,x_test,y_train,y_test=train_test_split(x_over,y_over,test_size=0.2)

np.random.seed(101)
from sklearn.model_selection import KFold
#import kfold

kfold=KFold(n_splits=5, shuffle=False)

#KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.

cross_validation = []
for train_index, test_index in kfold.split(x_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train_kf, x_test_kf = x_train[train_index], x_train[test_index]
    y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

    # building model

    model_cv = KNeighborsClassifier(n_neighbors=5)
    model_cv.fit(x_train_kf, y_train_kf)

    # taking accuracy
    acc = round(model_cv.score(x_test_kf, y_test_kf) * 100, 2)  # round( accuracy_score(y_test, y_pred) * 100, 2 )
    cross_validation.append(acc * 100)  # *100 means making percentage

cross_validation  # to estimate the skill of a machine learning model on unseen data.

np.mean(cross_validation),pd.Series(cross_validation).var()
# MEAN & VARIENCE for cross_validation
# it has low varience (<10)

from sklearn.metrics import accuracy_score

#Import Library for Logistic Regression
from sklearn.linear_model import LogisticRegression

#Initialize the Logistic Regression Classifier
logisreg = LogisticRegression()

#Train the model using Training Dataset
logisreg.fit(x_train, y_train)

# Prediction using test data
y_pred = logisreg.predict(x_test)

# Calculate Model accuracy by comparing y_test and y_pred
acc_logisreg = round( accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Logistic Regression model : ', acc_logisreg )

#Import Library for Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Initialize the Linear Discriminant Analysis Classifier
model = LinearDiscriminantAnalysis()

#Train the model using Training Dataset
model.fit(x_train, y_train)

# Prediction using test data
y_pred = model.predict(x_test)

# Calculate Model accuracy by comparing y_test and y_pred
acc_lda = round( accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Linear Discriminant Analysis Classifier: ', acc_lda )

#Import Library for Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Initialize the Gaussian Naive Bayes Classifier
model = GaussianNB()

#Train the model using Training Dataset
model.fit(x_train, y_train)

# Prediction using test data
y_pred = model.predict(x_test)

# Calculate Model accuracy by comparing y_test and y_pred
acc_ganb = round( accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Gaussian Naive Bayes : ', acc_ganb )

#Import Library for Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

#Initialize the Decision Tree Classifier
model = DecisionTreeClassifier()

#Train the model using Training Dataset
model.fit(x_train, y_train)

# Prediction using test data
y_pred = model.predict(x_test)

# Calculate Model accuracy by comparing y_test and y_pred
acc_dtree = round( accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of  Decision Tree Classifier : ', acc_dtree )

#Import Library for Random Forest
from sklearn.ensemble import RandomForestClassifier

#Initialize the Random Forest
model = RandomForestClassifier()

#Train the model using Training Dataset
model.fit(x_train, y_train)

# Prediction using test data
y_pred = model.predict(x_test)

# Calculate Model accuracy by comparing y_test and y_pred
acc_rf = round( accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of  Random Forest : ', acc_rf )

# Cretae a dataframe with all models and score

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Linear Discriminant Analysis','Naive Bayes', 'Decision Tree', 'Random Forest',
              'K - Nearest Neighbors'],
    'Score': [acc_logisreg, acc_lda, acc_ganb, acc_dtree, acc_rf,  acc]})

models.sort_values(by='Score', ascending=False)