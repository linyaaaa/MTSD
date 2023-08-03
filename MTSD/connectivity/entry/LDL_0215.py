'''
copula参数特征
！！！
此时的copula为多变量高斯函数

经检验copula计算1.0版本靠谱
'''

from copulas.multivariate.gaussian import GaussianMultivariate
import numpy as np
import random
import xlrd
import time
import os
import xlwt
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec



def My_Copula(data):
    #1.0版本
    gc = GaussianMultivariate()
    #print('data:',data)
    print(gc.__str__())
    copula_data = gc.fit(data)
    #print(gc.__str__())
    #print('Covariance为Gaussian Copula参数，但是还需要均值特征')
    result = gc.covariance
    #print(gc.__str__())


    return copula_data