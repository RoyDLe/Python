#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:39:19 2023

@author: roylehmann
"""

from pypfopt import risk_models, expected_returns, objective_functions, EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import os, requests
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn")


class Optimizer():
    def __init__(self):
        self.__ticker = []
        self.__gamma = 0
        self.__total_portfolio_value=0
        self.__constraints = []
        self.__allocation = {}
        self.__rf = 0
        self.__data = pd.DataFrame()
        self.__equity_mapper = {}
        self.__asset_mapper = {}
    
    def set_optimizer(self, *args, lookback = 3):
        self.__ticker = []
        for product in args:
            prod = self.__get_ticker(product)
            if not prod:
                self.__log(f'{product} not found')
            else:
                self.__ticker.append(prod)
        if len(self.__ticker) > 0:
            now = datetime.now()
            history = now - timedelta(lookback*365)
            frames = [yf.download(ticker, history, now)["Close"] for ticker in self.__ticker]
            prices = pd.concat(frames, axis = 1)
            prices.columns = [t_cap.upper() for t_cap in self.__ticker]
            self.__data = prices
            return True
        else:
            self.__log('N/A')
            return False
        
    def add_product(self, product, lookback = 3):
        prod = self.__get_ticker(product)
        if not prod:
            self.__log(f'{product} not found')
            return False
        elif prod in self.__ticker:
            self.__log(f'{product} already in array. Please choose something else.')
        else:
            self.__ticker.append(prod)
            now = datetime.now()
            history = now - timedelta(lookback*365)
            frame = yf.download(prod, history, now)["Adj Close"]
            merged_frames = pd.concat([self.__data, frame], axis = 1)
            merged_frames.columns = [t_cap.upper() for t_cap in self.__ticker]
            self.__data = merged_frames
        
            return self.__data
    
    def remove_product(self, ticker):
        if len(self.__ticker) > 0 and ticker in self.__ticker:
            self.__ticker.remove(ticker)
            self.__data.drop(ticker, inplace=True, axis=1)
            if ticker in self.__asset_mapper.keys():
                self.__asset_mapper.pop(ticker)
            if ticker in self.__equity_mapper.keys():
                self.__equity_mapper.pop(ticker)
            if list(self.__data.columns) == self.__ticker:
                self.__log(f'{ticker} correctly removed.')
                return self.__data
            else: 
                print(f'<< WARNING >>\nColumn name <<{self.__data.columns}>> and ticker ordering <<{self.__ticker}>> mismatch.')
        elif len(self.__ticker) == 0:
            self.__log('Ticker array is empty')
            return False
        else:
            self.__log(f'{ticker} not in array')
            return False
        
    def get_data(self):
        return self.__data

    def set_rf(self, rf):
        if rf > 0:
            self.__rf = rf
            return True
        else:
            self.__log('Nominal rate cannot be less than 0.')
            return False
        
    def set_gamma(self, gamma):
        self.__gamma = gamma
        
    def set_portfolio_value(self, value):
        if value > 0:
            self.__total_portfolio_value = value
            return True
        else:
            self.__log('Value cannot be negative.')
            return False
    
    def add_constraint(self, constraint):
        self.__constraints.append(constraint)
    
    def remove_constraint(self, pos):
        if len(self.__constraints) > 0:
            self.__constraints.pop(pos)
        else:
            self.__log('Constraint array is empty')
    
    def clear_constraints(self):
        self.__constraints = []
        
    def echo(self):
        self.__log(f'Ticker: {self.__ticker}\nRisk-free rate: {self.__rf}\nGamma: {self.__gamma}\nPortfolio value: {self.__total_portfolio_value}\nNumber of constraints: {len(self.__constraints)}')
        
    def get_allocation(self):
        if self.__allocation:
            text = "------------------------------------------\n"
            for key in dict(self.__allocation).keys():
                text += f'{key}: {self.__allocation[key] * 100}%\n'
                
            print(text)
            return self.__allocation
        else:
            self.__log("Run an optimization beforehand.")
    
    def get_equity_mapper(self):
        if self.__equity_mapper:
            text = "------------------------------------------\nEQUITY MAPPER\n"
            for key in dict(self.__equity_mapper).keys():
                text += f'{key}: {self.__equity_mapper[key]}\n'
                
            print(text)
            return self.__equity_mapper
        else:
            self.__log("Equity mapper is empty.")

    def get_asset_mapper(self):
        if self.__asset_mapper:
            text = "------------------------------------------\nASSET MAPPER\n"
            for key in dict(self.__asset_mapper).keys():
                text += f'{key}: {self.__asset_mapper[key]}\n'
                
            print(text)
            return self.__asset_mapper
        else:
            self.__log("Asset mapper is empty.")
    
    def save_data(self, filename):
        filename = f'{os.getcwd()}/{filename}.csv'
        self.__data.to_csv(filename)
        
    def clean_data(self, method = 'ffill'):
        self.__data.fillna(method = method, inplace = True)
        print(self.__data.isnull().sum())
        return True
        
    def max_sharpe(self):
        if not self.__data.empty and self.__total_portfolio_value >0:
            mu = expected_returns.mean_historical_return(self.__data)         
            S = risk_models.CovarianceShrinkage(self.__data).ledoit_wolf()   
            ef = EfficientFrontier(mu, S)
            for constraint in self.__constraints:
                ef.add_constraint(constraint)
            ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            latest_prices = get_latest_prices(self.__data)
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=self.__total_portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            self.__allocation = cleaned_weights
            self.__log(f'Discrete allocation: {allocation}\n------------------------------------------')
            self.__log("Funds remaining: Rs.{:.2f}".format(leftover) +'\n------------------------------------------')
            ef.portfolio_performance(verbose=True)
            return True
        elif self.__data.empty:
            self.__log('Use "set_optimizer" first.')
            return False
        else:
            self.__log('Determine a nonzero portfolio value for discrete allocation.')
            

    def min_var(self):
        if not self.__data.empty and self.__total_portfolio_value >0:
            mu = expected_returns.mean_historical_return(self.__data)
            S = risk_models.CovarianceShrinkage(self.__data).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            ef.add_objective(objective_functions.L2_reg, gamma=self.__gamma)
            for constraint in self.__constraints:
                ef.add_constraint(constraint)
            ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            latest_prices = get_latest_prices(self.__data)
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=self.__total_portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            self.__allocation = cleaned_weights
            self.__log(f'Discrete allocation: {allocation}\n------------------------------------------')
            self.__log("Funds remaining: Rs.{:.2f}".format(leftover) +'\n------------------------------------------')
            ef.portfolio_performance(verbose=True)
        elif self.__data.empty:
            self.__log('Use "set_optimizer" first.')
            return False
        else:
            self.__log('Determine a nonzero portfolio value for discrete allocation.')
    
    def efficient_risk(self, target_risk):
        if not self.__data.empty and self.__total_portfolio_value >0:
            mu = expected_returns.mean_historical_return(self.__data)
            S = risk_models.CovarianceShrinkage(self.__data).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            ef.efficient_risk(target_risk)
            cleaned_weights = ef.clean_weights()
            self.__log(f'Cleaned weights: {cleaned_weights}\n------------------------------------------')
            latest_prices = get_latest_prices(self.__data)
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=self.__total_portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            self.__allocation = allocation
            self.__log(f'Discrete allocation: {allocation}\n------------------------------------------')
            self.__log("Funds remaining: Rs.{:.2f}".format(leftover) +'\n------------------------------------------')
            ef.portfolio_performance(verbose=True)
        elif self.__data.empty:
            self.__log('Use "set_optimizer" first.')
        else:
            self.__log('Determine a nonzero portfolio value for discrete allocation.')
                
    def get_corr(self):
        if not self.__data.empty:
            pct_change_df = self.__data.pct_change()
            return pct_change_df.corr(min_periods = 50)
        else:
            self.__log('Dataframe empty.')
            return False
        
    def get_expected_return(self, product, returns = 'simple'):
        if product in self.__ticker and product in self.__data.columns:
            if returns == 'simple':
                pct_change_df = self.__data[product].pct_change()
                return f'Expected Annual Return: {round((pct_change_df.mean() * 252) * 100, 2)} %'
            elif returns == 'logs':
                #FIX
                log_return_df = np.log(self.__data[product]) - np.log(self.__data[product].shift(1))
                print(log_return_df)
                return f'Expected Annual Return: {round((log_return_df.mean() * 252) * 100, 2)} %'
            else:
                self.__log('Return type unknown.')
        else:
            self.__log(f'{product} not in array.')
    
    def get_portfolio_metrics(self, portfolio_value = 10000, rolling_window = 30):
        optimized_port_returns = self.__get_weighted_return_series()
        drop_index = optimized_port_returns.reset_index()
        
        normalized_series = []
        
        for index, row in optimized_port_returns.iterrows():
            portfolio_value = portfolio_value * (1+float(row))
            normalized_series.append(portfolio_value)
        
        df = pd.DataFrame(normalized_series, columns = ['Portfolio Value'])
        df.set_index(drop_index['Date'], inplace = True)     
    
        annualized_return = optimized_port_returns.mean() * 252
        annualized_risk = optimized_port_returns.std() * np.sqrt(252)
        annual_sharpe = (annualized_return - self.__rf) / annualized_risk
        
        rolling_mean = df.rolling(rolling_window).mean()
        
        rolling_std = optimized_port_returns.rolling(rolling_window).std() * np.sqrt(rolling_window)
        rolling_corr = optimized_port_returns.rolling(rolling_window).corr(rolling_std)
        
        fig = plt.figure()
        fig.suptitle(f'Rolling standard deviation (risk) last t-{rolling_window} days.')
        plt.plot(rolling_std)
        plt.xticks(rotation = 45)
        
        fig3 = plt.figure()
        fig3.suptitle(f'Rolling correlation of {rolling_window} day average return and risk.')
        plt.plot(rolling_corr)
        plt.xticks(rotation = 45)
        
        fig2 = plt.figure()
        fig2.suptitle(f'Returns and rolling average return last t-{rolling_window} days.')
        plt.plot(df['Portfolio Value'])
        plt.plot(rolling_mean)
        plt.xticks(rotation = 45)

        
        print(f'Annualized Average Return: {float(round(annualized_return * 100,2))}%\nAnnualized Risk: {float(round(annualized_risk*100,2))}%\nSharpe Ratio: {float(round(annual_sharpe,2))}')
    
    def get_efficient_frontier(self, sim_portfolios = 10000, seed = 1123):
        returns = self.__data.pct_change().dropna()
        noa = len(self.__data.columns)
        np.random.seed(seed)
        matrix = np.random.random(noa * sim_portfolios).reshape(sim_portfolios, noa)
        weights = matrix / matrix.sum(axis = 1, keepdims = True)
        
        port_ret = returns.dot(weights.T)
        
        summary = self.__ann_risk_return(returns)
        print(summary)
        port_summary = self.__ann_risk_return(port_ret)
        
        plt.figure()
        plt.scatter(port_summary.loc[:, 'Risk'], port_summary.loc[:, 'Return'], s=20, color = 'red')
        plt.scatter(summary.loc[:, 'Risk'], summary.loc[:, 'Return'], s=50, color = 'black', marker = 'D')
        plt.xlabel('Annual Risk')
        plt.ylabel('Annual Return')
        plt.title('Risk/Return')
        plt.show()

    def __ann_risk_return(self, df):
        summary = df.agg(['mean', 'std']).T
        summary.columns = ['Return', 'Risk']
        summary.Return = summary.Return * 252
        summary.Risk = summary.Risk*np.sqrt(252)
        
        return summary
        
    def __get_weighted_return_series(self):
        if self.__allocation:
            
            columnNum = len(self.__data.columns)
            columnNames = self.__data.columns
            
            #Check if ordering in dataframe matches ordering in portfolio dictionary to match weights to returns in dataframe
            for index in range(len(columnNames)):
                if columnNames[index] != self.__ticker[index]:
                    print('WARNING: Dataframe does not match dictionary ordering!')
                    break
            if columnNum == len(self.__allocation.keys()): #Dataframe is only allowed to contain price data for all stocks
                concatenatedReturns = self.__data.pct_change()
                drop_index = concatenatedReturns.reset_index()
                returnSeries = []
                
                for row in concatenatedReturns.itertuples():
                    dailyReturn = 0
                    for i in range(1,len(row)):
                        if not pd.isnull(row[i]):
                            dailyReturn += float(row[i]) * list(self.__allocation.values())[i-1]
                    returnSeries.append(dailyReturn)
                df = pd.DataFrame(returnSeries, columns = ['Portfolio Returns'])
                df.set_index(drop_index['Date'], inplace = True)
                return df 
        else:
            self.__log("Run an optimization beforehand.")

            
    def __get_ticker(self, 
                     product_name, 
                     max_tries = 5, 
                     allowed_exchanges = ["NMS", "AMS","PCX","NYQ","CMX", "GER", "BER", "LSE", "HAM", ],
                     yfinance_ = "https://query2.finance.yahoo.com/v1/finance/search",
                     user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'):
        if max_tries == 0:
            return False
        else:
            params = {"q": product_name, "quotes_count": 5}
            self.__log(f'Searching for: {product_name}')
            ticker = ""
        
            try:
                res = requests.get(url=yfinance_, params=params, headers={'User-Agent': user_agent})
                data = res.json()
                for exchange in allowed_exchanges:
                    for i in range(0, len(data['quotes'])):
                        if exchange == data['quotes'][i]['exchange']:
                            ticker = data['quotes'][i]['symbol']
                            """------------------------------------"""
                            asset_type = data['quotes'][i]['quoteType']
                            self.__asset_mapper[ticker] = asset_type
                            if asset_type.upper() == 'EQUITY':
                                self.__equity_mapper[ticker] = data['quotes'][i]['sector']
                            break
                    if ticker != "" and ticker != None:
                        break
                if ticker == "" or ticker == None:
                    product_name = product_name.rsplit(' ', 1)[0]
                    return self.__get_ticker(product_name, max_tries -1)
                else:
                    print(f'------------------------------------------\nTicker selected: {ticker}\n------------------------------------------')
                    return ticker
                
            except IndexError:
                product_name = product_name.rsplit(' ', 1)[0]
                return self.__get_ticker(product_name, max_tries - 1) 
            except KeyError:
                product_name = product_name.rsplit(' ', 1)[0]
                return self.__get_ticker(product_name, max_tries - 1) 
            
    def __log(self, text):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {text}')
        


