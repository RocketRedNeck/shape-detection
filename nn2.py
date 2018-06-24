# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:16:35 2018

@author: erkessel
"""
from numpy import exp, array, random, dot
import copy

training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
training_set_outputs = array([[0, 1, 1, 0, 1, 1, 0, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
initial_weights = copy.copy(synaptic_weights)
for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
    print("Iteration #" + str(iteration) + " weights: ", synaptic_weights)
print("---------------------------")

print("Initial weights: ")
print(initial_weights)
print("Final weights:   ")
print(synaptic_weights)
print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))