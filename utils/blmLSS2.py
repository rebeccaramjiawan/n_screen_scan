#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:18:24 2018

@author: fvelotti
"""

import numpy as np
import pandas as pd



class blmLSS2():

    def __init__(self, japcIn):

        self.japc = japcIn

    def getValue(self):
        
        losses_vec, info = self.japc.getParam("BLRSPS_LSS2/Acquisition#calLosses", getHeader=True)
        # losses_vec, info = self.japc.getParam("BLRSPS_LSS2/ExpertAcquisition#rawLosses", getHeader=True)
        
        channel_names = self.japc.getParam("BLRSPS_LSS2/Acquisition#channelNames")
        losses_vec = losses_vec[:len(channel_names)]
        
        self.losses_vec = losses_vec
        self.chNames = channel_names
        self.timeStamp = info['acqStamp']
        
    def get216(self, intExtr):
        losses_vec, info = self.japc.getParam("BLRSPS_BA2/Acquisition#calLosses", getHeader=True)
        channel_names = self.japc.getParam("BLRSPS_BA2/Acquisition#channelNames")
        
        target = 'SPS.BLM.216'
        ind216 = list(channel_names).index(target)
        
        self.timeStamp = info['acqStamp']
        
        return losses_vec[ind216] / intExtr
        
    
    def selectUseful(self):
        
        self.usefulBLM = self.chNames[:6]
        
    
    def getTotalLosses(self):
        
        self.getValue()
        self.selectUseful()
    
        df = pd.DataFrame(self.losses_vec, index=self.chNames, columns=['lossesBLM'])
        
        self.totalLosses = df['lossesBLM'][list(self.usefulBLM)].sum()
        
    def getTotalLossesNorm(self, intExtr):
        self.getTotalLosses()
        
        return self.totalLosses / intExtr
