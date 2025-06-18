#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:49:35 2022

@author: roylehmann
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from YDatamanager import YDatamanager
from Optimizer import Optimizer
from scipy.stats import norm
import seaborn as sns
import os
from datetime import datetime, timedelta

class Simulator():
    def __init__(self, portfolio, simulationId):
        self.__portfolio = portfolio # Not allowed to change portfolio from outside class as it can lead to inconsistencies
        self.__simulationId = simulationId
        self.filemanager = YDatamanager(portfolio, simulationId)
        self.optimizer = Optimizer(simulationId)
        self.__simulationOutput = pd.DataFrame()
        print(f'Portfolio correctness: {self.check_portfolio_correctness()}')
        
    def simulate(self, numSimulations, simulationTimeframe = 1, data_history = 10, include_average = True, plot_histogram = True):
        
        np.random.seed()
        df = pd.DataFrame()
        
        path = f'{os.getcwd()}/Simulation{self.__simulationId}.csv'
        if os.path.exists(path): 
            try:
                temp = pd.read_csv(path, index_col = 0)
                diff = (datetime.strptime(str(temp.index[-1]), '%Y-%m-%d')-datetime.strptime(str(temp.index[0]), '%Y-%m-%d'))/timedelta(days=365)
                if list(temp.columns) == list(self.__portfolio.keys()) and round(diff, 1) == data_history:
                    print('Required file exists')
                    df = temp
                else: df = self.filemanager.get_dataframe_from_yahoo(data_history)
            except ValueError as va: 
                df = self.filemanager.get_dataframe_from_yahoo(data_history)   
                print(va)
        else: df = self.filemanager.get_dataframe_from_yahoo(data_history) # If file does not exist or is wrong, download data from Yahoo
            
        returns = self.filemanager.get_weighted_return_series(df)
        
        # Perform operations on numpy array
        logReturns = np.log(1+returns) # Historical log returns
        if plot_histogram == True:
            sns.displot(logReturns.iloc[1:], kde = True)
            
        mean = logReturns.mean()
        var = logReturns.var()
        drift = mean - (0.5*var)
        std = logReturns.std()
        Z = norm.ppf(np.random.rand(simulationTimeframe*252, numSimulations))
        returns = np.exp(drift.values + std.values * Z)
        
        paths = np.zeros_like(returns)
        paths[0] = 1
        for t in range(1, simulationTimeframe*252):
            paths[t] = paths[t-1] * returns[t]
        
        df = pd.DataFrame(paths)
        if include_average == True:
            averageSeries = self.get_mean_series(df)
            self.__simulationOutput = pd.concat([df, averageSeries], axis = 1)
        else: 
            self.__simulationOutput = df
        
        return self.__simulationOutput
            
    def optimize_sharpe(self):
        optimum = self.optimizer.max_sharpe()
        self.__portfolio = optimum
        self.filemanager.overwrite_portfolio(optimum)
    
    def optimize_risk(self):
        optimum = self.optimizer.min_var()
        self.__portfolio = optimum
        self.filemanager.overwrite_portfolio(optimum)
        
    def check_portfolio_correctness(self):
        weights = [weight for weight in self.__portfolio.values()]
        if sum(weights) == 1 and sorted(weights)[0] >= 0:
            return True
        else: return False
    
    def set_portfolio(self, portfolio):
        self.__portfolio = portfolio 
        self.filemanager.overwrite_portfolio(portfolio)
        
    def plot(self, title='Portfolio Simulation', xLabel = 'T', yLabel='Normalized Portfolio Value'):
        if not self.__simulationOutput.empty:
            fig = plt.figure()
            fig.suptitle(title)
            plt.plot(self.__simulationOutput)
            if 'Average' in self.__simulationOutput.columns:
                plt.plot(self.__simulationOutput["Average"], linewidth=2, color='k')
            
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.show()
        
    def get_probability_of_outcome(self, benchmarkReturn):
        if not self.__simulationOutput.empty:
            finalRow = [row for row in self.__simulationOutput.itertuples()][-1]
            finalValues = [value for value in finalRow]
            finalValues.remove(finalRow.Index)
        
            higherThan = [ret-1 for ret in finalValues if ret-1 >= benchmarkReturn]
            lowerThan = [ret-1 for ret in finalValues if ret-1 < benchmarkReturn]
        
            probHigher = len(higherThan)/len(finalValues)
            probLower = len(lowerThan)/len(finalValues)
        
            return f'P(Return >= {benchmarkReturn}) = {probHigher}\nP(Return < {benchmarkReturn}) = {probLower}'
        else:
            return None
    
    def get_expected_return(self, plot_histogram = True):
        if not self.__simulationOutput.empty:
            pointEstimate = self.__simulationOutput.iloc[[-1]]['Average'].to_string(index = False)
            finalRow = self.__simulationOutput.iloc[-1]
            del finalRow['Average']
            
            if plot_histogram == True:
                sns.displot(finalRow, kde = True) #Plot histogram of final portfolio values
                plt.xlabel('Final Normalized Portfolio Values')
        
            return f'Expected return: {(float(pointEstimate)-1)*100}%'
        else:
            return None
    
    def get_mean_series(self, dataframe, label = 'Average'):
        cleanRows = []
        averages = []
        rows = [row for row in dataframe.itertuples()]
        for row in rows:
            rowValues = [value for value in row]
            rowValues.remove(row.Index)
            cleanRows.append(rowValues)
        for row in cleanRows:
            averages.append(np.mean(row))
        
        df = pd.DataFrame(averages, columns = [label])
        return df 
    
    def get_simulation_output(self):
        return self.__simulationOutput
    
    
        
            
            
            
            

        
        
            
        
        
        
        
        
              
        
        
    
        
        
            
        
        
        
    
        