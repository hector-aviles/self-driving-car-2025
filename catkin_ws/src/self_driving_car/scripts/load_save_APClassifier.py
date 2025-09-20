import sys
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import model_selection
from APClassifier import RuleClassifier
import pickle
import time
import statistics

def main():

    pl_fmdp = RuleClassifier()
          
    filename = "APClassifier_10.ap"
    pickle.dump(pl_fmdp, open(filename, 'wb'))  
    

if __name__ == "__main__":

    main()


