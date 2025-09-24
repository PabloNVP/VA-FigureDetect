from sklearn import tree
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os

class Trainer:
    def __init__(self):
        try:
          df = pd.read_csv('data.csv')
        except:
          print("No existe el archivo data.csv")
          exit(1)

    def pretraining(self):
        X = df.drop('label', axis=1).astype(float).values
        Y = df['label'].values

    def training(self):
        clasificador = tree.DecisionTreeClassifier(random_state=42).fit(X, Y)

    def show_model(self):
        plt.figure(figsize=(12,8))
        tree.plot_tree(
            clasificador, 
            filled=True, 
            feature_names=df.columns[:-1], 
            class_names=df['label'].unique(),
            rounded=True
        )
        plt.savefig("tree_plot.png", dpi=300)
        plt.close()

if __name__ == "__main__":
  tr = Trainer()
  tr.pretraining()
  tr.training()
  tr.show_model() 







