import os
import numpy as np
import pandas as pd
import keras
import requests
import csv
import matplotlib.pyplot as plt
from data_aggregator import getPandasDF
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
scalar = MinMaxScaler(feature_range=(0,1))

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


def createTestandTrainData(df):

    #scale the data between 0 and 1
    # note ranges in different time periods
    df = df['close']

    df = scalar.fit_transform(np.array(df).reshape(-1,1))
    training_size = int(len(df)*0.65)
    test_size=len(df)-training_size
    train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    return df, X_train, y_train, X_test, ytest

def createModel():
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def predict(model, X_train, X_test, y_train, ytest):
    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    #
    #Transformback to original form
    train_predict=scalar.inverse_transform(train_predict)
    test_predict=scalar.inverse_transform(test_predict)
    print("train RMSE")
    print(math.sqrt(mean_squared_error(y_train,train_predict)))
    print("test RMSE")
    print(math.sqrt(mean_squared_error(ytest,test_predict)))
    return train_predict, test_predict

def predictAddPredictions(model, X_train, X_test, y_train, ytest):
    train_predict = pd.DataFrame().reindex_like(y_train)
    
    train_predict[] = model.predict(X_train)


def plotResults(df, train_predict, test_predict):
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scalar.inverse_transform(df))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()



if __name__ == "__main__":
    #create test and train dataLabels
    df = getPandasDF("msft", ["high", "low", "open", "close", "volume"],
                                     "2020-10-10", "5min")
    close_df, X_train, y_train, X_test, ytest = createTestandTrainData(df)
    #run data on model
    model = createModel()

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=20,
              batch_size=64,verbose=1)
    train_predict, test_predict = predict(model, X_train, X_test, y_train, ytest)
    plotResults(close_df, train_predict, test_predict)
