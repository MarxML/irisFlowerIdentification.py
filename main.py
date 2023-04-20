#########################################################################################
# main.py                                                                               #
#                                                                                       #
# This is my first ML project, the "hello world" of machine learning. This program      #
# identifies iris flowers based on a dataset of 150 observations                        #
#                                                                                       #
# This is based on a tutorial by Jason Brownlee                                         #
#                                                                                       #
# Last edited 04/19/2023 by Adam Marx                                                   #
#########################################################################################

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Description 
print("\n___________________________________________________________")
print("    DESCRIPTION OF ATTRIBUTES:")
print("___________________________________________________________")
print(dataset.describe())

# Class distribution
print("___________________________________________________________")
print("    CLASS DISTRIBUTION:")
print("___________________________________________________________")
print(dataset.groupby('class').size())
print("___________________________________________________________\n\n")

#***********************************************************************************#
# The following section produces graphs to illustrate the spread of the given data.
# It's commented out by default, so you don't have to click through multiple 
# pop ups. If you want to re-enable them, un comment any line with a double hash '##'

# box and whisker plots
##dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
##pyplot.show()

# histograms
##dataset.hist()
##pyplot.show()

# scatter plot matrix
##scatter_matrix(dataset)
##pyplot.show()

#***********************************************************************************#

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('Lin. Disc. Analysis', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors', KNeighborsClassifier()))
models.append(('Class./Reg. Trees', DecisionTreeClassifier()))
models.append(('Gauss Naive Bayes', GaussianNB()))
models.append(('Support Vec. Machines', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
print("\n___________________________________________________________")
print("    ACCURACY OF EACH ALGORITHIM IN PERCENT:                ")
print("___________________________________________________________")
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%-37s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
 print("___________________________________________________________")

# Compare Algorithms
##pyplot.figure(figsize=(20,7))
##pyplot.boxplot(results, labels=names)
##pyplot.title('Algorithm Comparison')
##pyplot.show()

#******************************************************************************************#
# Based on the results above, we will use the Support Vector Machines (SVC) algorithm
# to make predictions

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
