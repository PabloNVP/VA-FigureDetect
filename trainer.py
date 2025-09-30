from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
    def __init__(self):
        try:
          self.df = pd.read_csv('./data/dataset.csv')
        except:
          print("No existe el archivo dataset.csv")
          exit(1)

    def pretraining(self):
        X = self.df.drop('label', axis=1).astype(float).values
        Y = self.df['label'].values
      
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

    def training(self) -> None:
        self.classifier = tree.DecisionTreeClassifier(random_state=42).fit(self.X_train, self.Y_train)

    def evaluate(self) -> None:
        if self.classifier is None:
            print("El modelo no está entrenado.")
            return
        
        y_pred = self.classifier.predict(self.X_test)
        labels = labels = self.classifier.classes_
        cm = confusion_matrix(self.Y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6,6))
        cax = ax.matshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                    va='center', ha='center',
                    color="black" if cm[i, j] < cm.max()/2 else "white")

        plt.savefig("./results/tree_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()


    def show_model(self):
        plt.figure(figsize=(12,8))
        tree.plot_tree(
            self.classifier, 
            filled=True, 
            feature_names=self.df.columns[:-1], 
            class_names=self.df['label'].unique(),
            rounded=True
        )
        plt.savefig("./results/tree_plot.png", dpi=300)
        plt.close()

    def save_model(self):
        joblib.dump(self.classifier, './models/tree_model.joblib')

if __name__ == "__main__":
    tr = Trainer()
    tr.pretraining()
    tr.training()
    tr.evaluate()
    tr.show_model()
    tr.save_model()
