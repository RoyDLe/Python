#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:24:23 2022

@author: roylehmann
"""
from pypfopt import risk_models, expected_returns, objective_functions, EfficientFrontier
import pandas as pd
import os

class Optimizer():
    def __init__(self, simulationId):
        self.__simulationId = simulationId
        self.rf = 0.0289
        self.gamma = 0
        self.constraints = []
    
    def max_sharpe(self, df = pd.DataFrame()): #Requires a dataframe of price data. Default = empty
        
        fail = False
        if df.empty: 
            direc = f'{os.getcwd()}/Simulation{self.__simulationId}.csv'
            if os.path.exists(direc):
                df = pd.read_csv(direc)
                del df['Date']
            else:
                print('No file found')
                fail = True
        else: del df['Date']
        
        if fail == False:
            mu = expected_returns.mean_historical_return(df)
            S = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu, S)
            if self.gamma != 0 and len(self.constraints) == 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.gamma)
            elif len(self.constraints) > 0:
                for constraint in self.constraints: 
                    ef.add_constraint(constraint)
            else: print('No optimization parameters detected.')
            weights = ef.max_sharpe(risk_free_rate = self.rf)
            return dict(weights)
        else: return None
    
    def min_var(self, df = pd.DataFrame()):
        fail = False
        if df.empty: 
            direc = f'{os.getcwd()}/Simulation{self.__simulationId}.csv'
            if os.path.exists(direc):
                df = pd.read_csv(direc)
                del df['Date']
            else:
                print('No file found')
                fail = True
        if fail == False: 
            mu = expected_returns.mean_historical_return(df)
            S = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu, S)
            if self.gamma != 0 and len(self.constraints) == 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.gamma)
            elif len(self.constraints) > 0:
                for constraint in self.constraints: 
                    ef.add_constraint(constraint)
            else: print('No optimization parameters detected.')
            weights = ef.min_volatility()
            return dict(weights)
        else: return None
    

        
    
    
            
        
            
                
            
            
        
            
            
            
        
        
        