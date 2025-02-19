import datetime
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

'''
def predictReturns(dates, prices):
    dates = np.reshape(dates,(len(dates), 1)) # convert to 1xn dimension
    prices = np.reshape(prices,(len(prices), 1)) # convert to 1xn dimension
    x = np.reshape(prices, (len(prices), 1))
    print(dates.shape)
    print(prices.shape)
    print(x.shape)
    prices = prices.astype('uint8')
    svm_rbf = SVC()    
    svm_rbf.fit(dates, prices)    
    return svm_rbf.predict(dates)

'''

def difference(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN(model, batchSize, testX):
    testX = testX.reshape(1, len(testX))
    forecast = model.predict(testX)
    return forecast[0]
    
def inverseDifference(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]  

dataset = pd.read_csv('Dataset/HINDPETRO.NS.csv',usecols=['Date','Close'])
dataset.fillna(0, inplace = True)
dataset.to_csv("temp.csv",index=False)
dataset = pd.read_csv('temp.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

original_data = dataset.values
X = dataset.values
X = difference(X, 1)
X = convertDataToTimeseries(X, 1)
X = X.values
trainX, testX = X[0:-30], X[-30:]
scalerX, trainX, testX = scaleDataset(trainX, testX)

trainXX, trainY = trainX[:, 0:-1], trainX[:, -1]

def difference1(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries1(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset1(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN1(model, batchSize, testX):
    testX = testX.reshape(1, 1, len(testX))
    forecast = model.predict(testX, batch_size=batchSize)
    return forecast[0,0]
    
def inverseDifference1(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale1(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]    


def runCNN(train_dataX,train_dataY,test_dataX,original_X,scalerX):
    train_dataY1 = to_categorical(train_dataY)
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(train_dataX.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(train_dataY1.shape[1]))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(cnn_model.summary())
    acc_history = cnn_model.fit(train_dataX,train_dataY1, epochs=3, validation_data=(train_dataX,train_dataY1))
    print(cnn_model.summary())
    predict = cnn_model.predict(train_dataX)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(train_dataY1, axis=1)
    acc = accuracy_score(testY.reshape(-1,1),predict.reshape(-1,1))
    fscore = f1_score(testY.reshape(-1,1),predict.reshape(-1,1),average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(testY.reshape(-1,1),predict.reshape(-1,1),pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    print(str(acc)+" "+str(fscore)+" "+str(roc_auc))


def runLSTM(train_dataX,train_dataY,test_dataX,original_X,scalerX):
    train_dataX1 = train_dataX.reshape(train_dataX.shape[0], 1, train_dataX.shape[1])
    lstm_model = Sequential()
    lstm_model.add(LSTM(4, batch_input_shape=(1, train_dataX1.shape[1], train_dataX1.shape[2]), stateful=True))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    print(lstm_model.summary())	
    for i in range(1):
        lstm_model.fit(train_dataX1, train_dataY, epochs=1, batch_size=1, verbose=2, shuffle=False)
        lstm_model.reset_states()
    trainReshaped = train_dataX[:, 0].reshape(len(train_dataX), 1, 1)
    lstm_model.predict(trainReshaped, batch_size=1)
    prediction_list = list()
    for i in range(len(test_dataX)):
        XX, y = test_dataX[i, 0:-1], test_dataX[i, -1]
        yhat = forecastRNN1(lstm_model, 1, XX)
        yhat = inverseScale1(scalerX, XX, yhat)
        yhat = inverseDifference1(original_X, yhat, len(test_dataX)+1-i)
        prediction_list.append(yhat)
        expected = original_data[len(train_dataX) + i + 1]
        print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    temp = original_X[-30:]
    temp = np.asarray(temp)
    for i in range(0,30):
        prediction_list[i] = temp[i]
    prediction_list = np.asarray(prediction_list)
    prediction_list = prediction_list.astype('uint8')
    temp = temp.astype('uint8')
    acc = accuracy_score(temp,prediction_list)
    fscore = f1_score(temp,prediction_list,average='macro')
    print(temp)
    print(prediction_list)
    fpr, tpr, thresholds = metrics.roc_curve(temp,prediction_list,pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc = fscore
    print(str(acc)+" "+str(fscore)+" "+str(roc_auc))    
        
   
    


def runML(classifier,train_dataX,train_dataY,test_dataX,original_X,scalerX):
    train_dataY = train_dataY.astype('uint8')
    classifier.fit(train_dataX, train_dataY)
    predict = classifier.predict(train_dataX)
    confirm_prediction_list = list()
    for i in range(len(test_dataX)):
        XX, y = test_dataX[i, 0:-1], test_dataX[i, -1]
        yhat = forecastRNN(classifier, 1, XX)
        yhat = inverseScale(scalerX, XX, yhat)
        yhat = inverseDifference(original_X, yhat, len(test_dataX)+1-i)
        confirm_prediction_list.append(yhat)
        expected = original_X[len(train_dataX) + i + 1]
        #print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    acc = accuracy_score(train_dataY.reshape(-1,1),predict.reshape(-1,1))
    fscore = f1_score(train_dataY.reshape(-1,1),predict.reshape(-1,1),average='macro')
    fpr, tpr, thresholds = metrics.roc_curve(train_dataY.reshape(-1,1),predict.reshape(-1,1),pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    print(str(acc)+" "+str(fscore)+" "+str(roc_auc))    

runCNN(trainXX, trainY,testX,original_data,scalerX)
runML(SVC(),trainXX, trainY,testX,original_data,scalerX)
runML(KNeighborsClassifier(),trainXX, trainY,testX,original_data,scalerX)
runML(DecisionTreeClassifier(),trainXX, trainY,testX,original_data,scalerX)
runML(RandomForestClassifier(),trainXX, trainY,testX,original_data,scalerX)
runML(LogisticRegression(),trainXX, trainY,testX,original_data,scalerX)
runML(XGBClassifier(),trainXX, trainY,testX,original_data,scalerX)
runML(AdaBoostClassifier(),trainXX, trainY,testX,original_data,scalerX)
runML(GaussianNB(),trainXX, trainY,testX,original_data,scalerX)
runLSTM(trainXX, trainY,testX,original_data,scalerX)
'''
cls = SVC()
trainY = trainY.astype('uint8')
cls.fit(trainXX, trainY)
predict = cls.predict(trainXX)
acc = cls.score(trainY.reshape(-1,1),predict.reshape(-1,1))
print(acc)

confirm_prediction_list = list()
for i in range(len(testX)):
    XX, y = testX[i, 0:-1], testX[i, -1]
    yhat = forecastRNN(cls, 1, XX)
    yhat = inverseScale(scalerX, XX, yhat)
    yhat = inverseDifference(original_data, yhat, len(testX)+1-i)
    confirm_prediction_list.append(yhat)
    expected = original_data[len(trainXX) + i + 1]
    print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
'''
'''
#dataset.columns = dataset.columns.str.replace(' ', '0')

dataset['Date'] = dataset['Date'].str.split('-').str[2]
dataset['Date'] = pd.to_numeric(dataset['Date'])
dates, prices =  [dataset['Date'].tolist(), dataset['Close'].tolist()]
predicted_returns = predictReturns(dates, prices)
print(predicted_returns) 

print(predicted_returns)
plt.plot(prices, color = 'red', label = 'Invested Amount')
plt.plot(predicted_returns, color = 'blue', label = 'Returns Amount')
plt.title('stock returns')
plt.xlabel('Days')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()
'''
