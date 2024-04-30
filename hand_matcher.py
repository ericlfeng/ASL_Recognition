import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class HandMatcher:
    def __init__(self):
        df = pd.read_csv("data/cleanhands.csv")
        all = df.loc[:, df.columns != "label"]
        selected = []
        to_add = len(all)
        y_train = df["label"]
        scorelist = []

        model = KNeighborsClassifier(n_neighbors=5)
        # Perform forward selection
        while to_add > 0:
            best_feature = None
            best_score = 0
            for feature in all:
                if feature not in selected:
                    combined_features = selected + [feature]
                    X_train, X_test, y_train, y_test = train_test_split(df[combined_features], df["label"], test_size= 0.2, random_state=5)
                    #print(combined_features)
                    model.fit(X_train, y_train)
                    model_score = model.score(X_test, y_test)
                    #print("Features:", selected, "+", feature + ", Score:", model_score)
                    if model_score > best_score:
                        best_score = model_score
                        best_feature = feature
            
            # Append best feature to model
            if(not best_feature):
                break
            
            
            selected.append(best_feature)
            print("Added best feature:", best_feature + ", Current feature list:", selected, "Current Score:", best_score)
            scorelist.append(best_score)
            to_add -= 1
        # Print final feature score
        X_train, X_test, y_train, y_test = train_test_split(df[combined_features], df["label"], test_size= 0.2, random_state=5)
        model.fit(X_train, y_train)
        initial_score = model.score(X_test, y_test)

        #print("Final features:", selected, "Score:", initial_score)


        #TODO: Save the weights of the model to a file so I can access it
        #In another file and easily remake the model

                
