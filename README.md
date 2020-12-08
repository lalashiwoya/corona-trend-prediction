<h1 align="center">
  Corona Trend Prediction
</h1>

## Our Goal
In our project, we built a LSTM model to predict corona trend in three countries: USA, UK and Germany and prepare the performance of using different features, i.e. stock index, new confirmed case and sentiment.

## What we have done
1. Data Preparation
   - Collected the number of new confirmed cases from USA, UK and Germany in the time period from Feb.20 until Nov.20 from [RKI] (https://www.arcgis.com/home/item.html?id=f10774f1c63e40168479a1feb6c7ca74).
   - Collected high price information of the contracts from [MCX](https://finance.yahoo.com/quote/MCX/history?p=MCX), [DJI] (https://finance.yahoo.com/quote/%5EDJI?p=^DJI&.tsrc=fin-srch), [DAX](https://finance.yahoo.com/quote/%5EGDAXI?p=^GDAXI&.tsrc=fin-srch).
   - Collected tweets related to refugee crisis and euro crisis in EU in the time period from Feb.20 until Nov.20.
   - Used [Vader](https://github.com/cjhutto/vaderSentiment) and [Flair](https://github.com/flairNLP/flair) to analyse the crawled english tweets.

2. Used moving average to smooth the kurve of number of new confirmed case (smooth=3 days & smooth=5 days).

3. Built a LSTM model 

## Running the python file lstm_corona.py and specify parameters like time span and features you are going to use to do the prediction. 

  <img src="https://github.com/lalashiwoya/corona-trend-prediction/blob/main/images/parameters.PNG" width=500>
   
