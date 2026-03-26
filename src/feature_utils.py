import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
import os
import sys


def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['MPWR', 'AAPL']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    Y = stk_data.loc[:, ('Adj Close', 'AAPL')]
    Y.name = 'AAPL'
    
    X = stk_data.loc[:, ('Adj Close', 'MPWR')]
    X.name = 'MPWR'
    
    dataset = pd.concat([Y, X], axis=1).dropna()
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.name]
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    return features


def extract_features_pair():
    """
    Downloads and prepares the pair data for NVDA & AVGO.
    Returns a DataFrame with two columns: [AVGO, NVDA]
    (partner first, target second — matching the notebook's data_prediction layout).
    """

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['AVGO', 'NVDA']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    # Partner column (AVGO) first, Target column (NVDA) second
    partner = stk_data.loc[:, ('Adj Close', 'AVGO')]
    partner.name = 'AVGO'

    target = stk_data.loc[:, ('Adj Close', 'NVDA')]
    target.name = 'NVDA'

    # Partner first, target second (same order as notebook's data_prediction)
    dataset = pd.concat([partner, target], axis=1).dropna()
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    return features


def get_bitcoin_historical_prices(days=60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df
