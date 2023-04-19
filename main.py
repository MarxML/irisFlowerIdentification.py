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


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
