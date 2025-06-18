#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:19:16 2023

@author: roylehmann
"""

from Optimizer import Optimizer

opt = Optimizer()
opt.set_optimizer('GC=F', "URTH", lookback = 3)
opt.clean_data()
opt.save_data("FTSE")
opt.set_portfolio_value(10000)
opt.min_var()
opt.get_efficient_frontier()
opt.get_allocation()
"""
opt.clean_data()
opt.predict_returns()

opt.add_constraint(lambda x: x[2] + x[3] == 0.05)
opt.add_constraint(lambda x: x>=0.025)
opt.set_portfolio_value(1000)
opt.set_gamma(0.02)
opt.min_var()
opt.get_efficient_frontier()
opt.get_asset_mapper()
opt.get_equity_mapper()

opt.get_allocation()

opt.max_sharpe()
opt.get_efficient_frontier()
opt.get_allocation()

"""
