#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:58:37 2022

@author: roylehmann
"""

from Simulator import Simulator


portfolio = {'ABR.DE': 0.8, 'AAPL': 0.2}

sim = Simulator(portfolio, 1)

result = sim.simulate(10000, data_history = 2, include_average=True)
sim.plot()

print(sim.get_expected_return())
print(sim.get_probability_of_outcome(0.0))


b = lambda x : x >= 0.01
sim.optimizer.constraints = [b]
sim.optimize_sharpe()

result = sim.simulate(10000, data_history = 2, include_average=True)
sim.plot()

sim.filemanager.to_csv(result)

print(sim.get_expected_return())
print(sim.get_probability_of_outcome(0.0))


















