#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:12:15 2019

@author: raghav
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train():
    df_train=pd.read_csv('training.csv')

    df_train['Image'] = df_train['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df_train.dropna(inplace=True)
    df_train.shape

    X = np.vstack(df_train['Image'].values)/255
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)

    y = df_train[df_train.columns[:-1]].values
    y = (y - 48) / 48  # scale target coordinates to [-1, 1] (Normalizing)
    y = y.astype(np.float32)
    
    return X,y

def test():
    df_test=pd.read_csv('test.csv')

    df_test['Image'] = df_test['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df_test.dropna(inplace=True)
    df_test.shape

    X = np.vstack(df_test['Image'].values)/255
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)
    
    return X
