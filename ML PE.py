# -*- coding: utf-8 -*-
"""

@author: Mareike Thurau
@author: Silvan Mueller

"""
import pandas as pd
import numpy as np

from math import exp
from random import random

# einlesen der Daten
trainingData = pd.read_csv("../data_u_1_3001879_5279494.csv")

class determineWeightCategory():

    def __init__ (self):
    
        self.MAX_INPUT_LAYER_SIZE=20
        self.MAX_HIDDEN_LAYER_SIZE=40
        self.MAX_OUTPUT_LAYER_SIZE=20

        self.INPUT_TO_HIDDEN=0
        self.HIDDEN_TO_OUTPUT=1  

        self.DEFAULT_EPSILON=1
        self.DEFAULT_LEARNING_RATE=0.5    
    
        self.inNeurons = 0
        self.hiddenNeurons = 0
        self.outNeurons = 0
    
