# -*- coding: utf-8 -*-
import numpy as np

def calculateMaxDD(cumRet):
# =============================================================================
# calculation of maximum drawdown and maximum drawdown duration based on
# cumulative COMPOUNDED returns. cumRet must be a compounded cumulative return.
# i is the index of the day with maxDD.
# =============================================================================
    highwatermark=np.zeros(cumRet.shape)
    drawdown=np.zeros(cumRet.shape)
    drawdownduration=np.zeros(cumRet.shape)
    
    for t in np.arange(1, cumRet.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumRet[t])
        drawdown[t]=(1+cumRet[t])/(1+highwatermark[t])-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
             
    maxDD, i=np.min(drawdown), np.argmin(drawdown) # drawdown < 0 always
    maxDDD=np.max(drawdownduration)
    return maxDD, maxDDD, i