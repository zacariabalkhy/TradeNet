import os
import numpy as np
import pandas as pd
import requests
import csv
import matplotlib.pyplot as plt


apiKey = "6bd0f2a0e3ef335196735529de663830884afc61"
partialRequestString = "https://api.tiingo.com/iex/"

def getPriceDataFromTiingo(ticker, metrics, date, sampleFrequency, format = "csv"):
    headers = {
        'Content-Type' : 'application/json'
    }
    fullRequestString = partialRequestString + ticker
    fullRequestString += "/prices?startDate=" + date + "&"
    fullRequestString += "resampleFreq=" + sampleFrequency + "&"

    fullRequestString += "columns="
    for metric in metrics:
        fullRequestString += metric
        if (metric != metrics[len(metrics) - 1]):
            fullRequestString += ","
        else:
            fullRequestString += "&"

    fullRequestString += "format=" + format + "&"
    fullRequestString += "token=" + apiKey
    requestResponse = requests.get(fullRequestString, headers=headers)
    return requestResponse

def formatData(csvData):
    contentList = []
    content = csvData.content.decode('utf-8')
    cr = csv.reader(content.splitlines(), delimiter=',')
    for row in cr:
        contentList.append(row)
    dataLabels = contentList[0]
    contentList = contentList[1:]
    df = pd.DataFrame(np.array(contentList).reshape(len(contentList),
                      len(contentList[0])), columns = dataLabels)
    df[["high", "low", "open", "close", "volume"]] = df[["high", "low", "open", "close", "volume"]].apply(pd.to_numeric)
    return df

def plotData(df):
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['low']+df['high'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

def getPandasDF(ticker, metrics, date, sampleFrequency, format = "csv"):
    csvData = getPriceDataFromTiingo(ticker, metrics, date, sampleFrequency, format)
    df = formatData(csvData)
    return df

if (__name__ == "__main__"):
    csvData = getPriceDataFromTiingo("gme", ["high", "low", "open", "close", "volume"],
                                     "2020-10-10", "5min")
    df = formatData(csvData)
    #plotData(df)
