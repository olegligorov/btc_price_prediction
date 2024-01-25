import quandl
import investpy
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas_ta as ta
from datetime import datetime,timedelta
from sklearn.preprocessing import RobustScaler,MinMaxScaler
import pickle
from tiingo import TiingoClient


def pred():
    yesterday_date = (datetime.today()-timedelta(days=1)).strftime('%d/%m/%Y')

    config = {}
    config['session'] = True
    config['api_key'] = "54f0e003492726e27cb11b03d6829cd099008623"
    client = TiingoClient(config)
    historical_prices = client.get_ticker_price("btcusd",
                                            fmt='json',
                                            startDate='2012-12-31',
                                            endDate=yesterday_date,
                                            frequency='daily')
    
    final_df = pd.DataFrame(historical_prices)
    final_df['date'] = pd.to_datetime(final_df['date']).dt.date
    final_df['date'] = final_df['date'].astype('datetime64')
    final_df.reset_index()
    final_df = final_df[['date','open','high','low','close']].round(2)
    final_df.columns = ['Date', 'opening_price', 'highest_price', 'lowest_price', 'closing_price']
    
    final_df['wma7 closing_price'] = ta.wma(final_df['closing_price'], 7)
    final_df['dema7 highest_price'] = ta.dema(final_df['closing_price'], 7)
    final_df['dema7 opening_price'] = ta.dema(final_df['highest_price'], 7)
    final_df['tema30 closing_price'] = ta.dema(final_df['closing_price'], 30)
    final_df['sma7 highest_price'] = ta.sma(final_df['highest_price'], 7)
    final_df['ema7 closing_price'] = ta.ema(final_df['closing_price'], 7)
    final_df['dema7 lowest_price'] = ta.dema(final_df['lowest_price'], 7)
    final_df['tema7 closing_price'] = ta.tema(final_df['closing_price'], 7)
    final_df = final_df[(final_df['Date'] >= '2013-01-01')].fillna(method='bfill')

    final_df = final_df[['Date', 'lowest_price', 'wma7 closing_price',
                        'dema7 highest_price', 'dema7 opening_price', 'tema30 closing_price',
                        'sma7 highest_price', 'ema7 closing_price',
                        'dema7 lowest_price', 'highest_price', 'tema7 closing_price']]

    sgd_reg = pickle.load(open('../LinearRegressionModel/linear_reg_10_35.sav', 'rb'))

    X = final_df.drop(['Date'],axis=1)

    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.fit_transform(X[X.columns])

    scaler = MinMaxScaler()
    X_scaled[X.columns] =  scaler.fit_transform(X_scaled[X.columns])

    today_btc_closing_price =  sgd_reg.predict(X_scaled.values[-1].reshape(1,-1))
    final_df['Date'] = final_df['Date'].dt.strftime('%d-%m-%Y')
    return final_df,round(float(today_btc_closing_price))

final_df, predicted_closing_price = pred()

print(f"The predicted BTC closing price for {datetime.today().strftime('%d/%m/%Y')} is {predicted_closing_price}$")