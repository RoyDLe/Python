#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:14:52 2022

@author: roylehmann
"""

import pandas as pd
from pandas_datareader import data
from datetime import datetime, timedelta
import os 

class YDatamanager():
    
    def __init__(self, portfolio, id):
        self.__portfolio = portfolio
        self.__tickers = [ticker for ticker in portfolio.keys()]
        self.__id = id
    
    def get_dataframe_from_yahoo(self, timeFrame):
        df = pd.DataFrame()
        now = datetime.now()
        history = now - timedelta(timeFrame*365)
        df = data.DataReader(self.__tickers, "yahoo", history, now)

        frames = [df['Adj Close'][ticker] for ticker in self.__tickers]
        df = pd.concat(frames, axis = 1)
        filename = f'{os.getcwd()}/Simulation{self.__id}.csv'
        df.to_csv(filename) #Cache
                
        return df
        
    def get_weighted_return_series(self, dataframe):
        columnNum = len(dataframe.columns)
        columnNames = dataframe.columns
        
        #Check if ordering in dataframe matches ordering in portfolio dictionary to match weights to returns in dataframe
        for index in range(len(columnNames)):
            if columnNames[index] != self.__tickers[index]:
                print('WARNING: Dataframe does not match dictionary ordering!')
                break
        if columnNum == len(self.__portfolio.keys()): #Dataframe is only allowed to contain price data for all stocks
            concatenatedReturns = dataframe.pct_change()
            returnSeries = []
            for row in concatenatedReturns.itertuples():
                dailyReturn = 0
                for i in range(1,len(row)):
                    if not pd.isnull(row[i]):
                        dailyReturn += float(row[i]) * list(self.__portfolio.values())[i-1]
                returnSeries.append(dailyReturn)
            df = pd.DataFrame(returnSeries, columns = ['Portfolio Returns'])
            return df
                        
        else:
            print('Remove non-price data from dataframe before performing this operation')
            return None
        
    def to_csv(self, dataframe):
        directory = os.getcwd()
        files = os.listdir(directory)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        sim_files = [file for file in files if f'Simulation{self.__id}' in file] 
        filename = f'{os.getcwd()}/Simulation{self.__id}.{len(sim_files) + 1}.csv'
        dataframe.to_csv(filename)                   
    
    def overwrite_portfolio(self, newPortfolio):
        self.__init__(newPortfolio, self.__id)
        
        

    
        
        
        
    
    
        
    

