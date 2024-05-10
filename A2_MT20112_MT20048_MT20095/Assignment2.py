import pandas as pd
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score
import sys

# map character to number (peptides encoding)
def charToNumEncoding(peptides):
    i = 0
    for character in peptides:
        mapping[character] = i
        i += 1


# Creating list of list of instances of training data
def traindataEncoding(data):
    Features = []
    row = len(data)
    for i in range(row):
        l = []
        sequence = data['Sequence'][i]
        for character in sequence:
            l.append(mapping[character])
        l.append(data['label'][i])
        Features.append(l)
    return Features


# Creating list of list of instances of testing data
def testdataEncoding(data):
    Features = []
    row = len(data)
    for i in range(row):
        l = []
        sequence = data['Sequence'][i]
        for character in sequence:
            l.append(mapping[character])
        Features.append(l)
    return Features



# data imbalance handling using oversampling on minority class
def Oversampling(x_train,y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    #print("x-shape :",X_resampled.shape,"\ny-shape :", y_resampled.shape)
    return X_resampled,y_resampled


# Applying RandomForest Classifier
def RandomForest(x,y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    y_pred = clf.predict(test_df)
    return y_pred     # returning predicted dependent variables

def XGBoost(x,y):
    clf = XGBClassifier()
    clf.fit(x, y)
    y_pred = clf.predict(test_df)
    return y_pred     # returning predicted dependent variables



# pipelining of RandomForest and XGBoost
def Pipelining(x,y):
    model = [('lr', RandomForestClassifier()), ('XG', make_pipeline(StandardScaler(), XGBClassifier()))]
    clf = StackingClassifier(estimators=model)
    clf.fit(x, y)
    y_pred = clf.predict(test_df)
    return y_pred    # returning predicted dependent variables

def GenerateCSV(y_pred):
    df3 = pd.read_csv('D:/2nd Semester/BDMH/Assignment/Assignment 2/BDMH_Assignment2_Dataset/test.csv')
    df3['label'] = y_pred
    compress = dict(method='zip', archive_name='result.csv')
    df3.to_csv('result.zip', index=False, compression=compress)


if __name__ == '__main__':
    trainPath = sys.argv[1]
    testPath = sys.argv[2]
    #df_train = pd.read_csv('D:/2nd Semester/BDMH/Assignment/Assignment 2/BDMH_Assignment2_Dataset/RNA_Train.csv')
    #df_test = pd.read_csv('D:/2nd Semester/BDMH/Assignment/Assignment 2/BDMH_Assignment2_Dataset/test.csv')
    df_train = pd.read_csv(trainPath) # D:/train.csv
    df_test = pd.read_csv(testPath)     # D:/test.csv"
    mapping = {}      # dictionary where key = character and value = numeric
    charToNumEncoding('XMLQVRAGTWFPSCNEKHIYD')     # RNA Sequence
    train_df = pd.DataFrame(traindataEncoding(df_train))
    test_df =  pd.DataFrame(testdataEncoding(df_test))
    x_train = train_df.loc[:, :16]       # independent variable
    y_train = train_df[17]               # dependent variable
    x_resampled, y_resampled = Oversampling(x_train,y_train)
    while(1):
        print("For RandomForest: enter 1")
        print("For XGBoost: enter 2")
        print("For Hybride model(RandomForest + XGBoost): enter 3\n")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            y_pred = RandomForest(x_resampled, y_resampled)
            GenerateCSV(y_pred)
        elif choice == 2:
            y_pred = XGBoost(x_resampled, y_resampled)
            GenerateCSV(y_pred)
        elif choice == 3:
            y_pred = Pipelining(x_resampled, y_resampled)
            GenerateCSV(y_pred)
        else:
            print("Wrong input")

        ch = input("\nTo continue press Y else press N : ")
        if ch == 'N':
            break









