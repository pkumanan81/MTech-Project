#import libraries 
import pandas as objpandas
import numpy as objnumpy
import matplotlib.pyplot as objplt
objplt.style.use('ggplot')
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from xgboost import XGBClassifier
from sklearn import metrics
import csv
import os

#XGBoost Algorithm Model Starts
#Load the data-set
dataset = objpandas.read_csv('D:/uploads/IoMTDataset.csv') 
#Print the count of rows and coulmns in csv file
print("Dimensions of Dataset: {}".format(dataset.shape))
# Dropped all the Null, Empty, NA values from csv file 
new_dataset = dataset.dropna(axis=0, how='any') 
print("Dimensions of Dataset after Pre-processing : {}".format(new_dataset.shape))
X = new_dataset.iloc[:, 0:28].values
print(X)
print("Dimensions of X : {}".format(X.shape))
y = new_dataset.iloc[:, 28:29].values
print("Dimensions of y : {}".format(y.shape))
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training and 30% test
#Create a XGBClassifier
model = XGBClassifier()
# fit model
model.fit(X_train, y_train)

#========================================================================================

# make a prediction
# data = [[1057, 618, 1484, 977.039, 143.23375, 506.727562, 311, 66, 60, 60, 4.560721, 1, 15, 1675, 2461, 0, 0, 0, 0, 3.07, 28.7, 98, 101, 138, 78, 84, 0, 0.4]];
#data = [[496, 186,	241147,	80382,	4.113667,	2.6895,	310, 66, 60, 60, 0.012341, 1, 7, 682, 321530, 0, 0, 0,0, 486.184, 28.7, 98,90, 138, 78, 88, 15, 0.26]];
# data = [[1057, 618, 1484, 977.039, 143.23375, 506.727562, 311, 66, 60, 60, 4.560721, 1, 15, 1675, 2461, 0, 0, 0, 0, 3.07, 28.7, 98, 101, 138, 78, 84, 0, 0.4]];
# data = [[436, 486, 520.952, 762.63, 46.762, 638.366, 310, 66, 60, 60, 4.468744, 1, 11, 922, 1284, 0, 0, 0, 0, 2.238, 29.1, 98, 87, 138, 78, 84, 0, 0.4]];
# data = [[556, 246, 12046, 5008, 73.8865, 71.353336, 310, 66, 60, 60, 0.295546, 1, 9, 802, 17053, 0, 0, 0, 0, 27.069, 27, 98, 82, 144, 78, 75, 0, 0.14]];

#print("Row : {}".format(data));

#y_pred = model.predict(data)

#iix = y_pred[0];

# summarize prediction
#print("Prediction Result : {}".format(y_pred[0]))

#results = 'Abnormal'

#if classes[iix] == 1:
 #   results = 'Abnormal'
#else:
 #   results = 'Normal'
  
#print(f'predicted class = {results}')

#========================================================================================
    
y_pred_1 = model.predict(X_test)
accuracy = model.score(X_test, y_pred_1);
print('Actual Acuracy::', accuracy)
r2 = metrics.r2_score(y_test, y_pred_1)
print('Actual R2 Score::', r2)
mae = metrics.mean_absolute_error(y_test, y_pred_1)
print('Actual Mean Absolute::', mae)
mse = objnumpy.sqrt(metrics.mean_squared_error(y_test, y_pred_1))
print('Actual Mean Square:', mse)

file = 'D:/uploads/output3.csv'

a1 = []
a2 = []
a3 = []
a4 = []

with open(file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        elif line_count == 1:
            a1 = [row[0], row[1], accuracy]
            line_count += 1
        elif line_count == 2:
            a2 = [row[0], row[1], r2]
            line_count += 1
        elif line_count == 3:
            a3 = [row[0], row[1], mae]
            line_count += 1
        else:
            a4 = [row[0], row[1], mse]
            line_count += 1

    print(f'Processed {line_count} lines.')
    
header = ['Sno', 'Linear Regression Classifier', 'XGB Classifier']
    
with open(file, 'w', newline='', encoding='UTF8') as csv_file1:
    objwriter = csv.writer(csv_file1)
    objwriter.writerow(header)
    objwriter.writerow(a1)
    objwriter.writerow(a2)
    objwriter.writerow(a3)
    objwriter.writerow(a4)

from sklearn.metrics import roc_curve, auc

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_1)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
objplt.figure(figsize=(8, 6))
objplt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
objplt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')
objplt.xlim([0.0, 1.0])
objplt.ylim([0.0, 1.05])
objplt.xlabel('False Positive Rate')
objplt.ylabel('True Positive Rate')
objplt.title('Receiver Operating Characteristic (ROC) Curve')
objplt.legend(loc="lower right")
objplt.savefig('D:/uploads/XGBoost_RNN_ROC_CURVE.png', dpi=1200)
objplt.show()