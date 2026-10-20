import numpy as np
from src.evaluate import choose_threshold, metrics

def test_perfect_predictions():
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
    m=metrics(y,p,.5)
    assert m["fraud_precision"]==1.0 and m["fraud_recall"]==1.0

def test_threshold_range():
    assert 0 <= choose_threshold(np.array([0,0,1,1]),np.array([.1,.2,.8,.9])) <= 1
