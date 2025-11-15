
MODELS_DIR = "./pickle_field"
from sklearn.metrics import accuracy_score, f1_score
from concrete.ml.sklearn import DecisionTreeClassifier as FHEDecisionTreeClassifier, RandomForestClassifier as FHERandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from itertools import product
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
import time
import pandas as pd
from multiprocessing import Pool

 
def experiment(modelname: str):
    max_depths = range(3, 15)
    n_bits = range(2, 12)
    n_features = range(5, 20)
    stats = pd.DataFrame(columns=["model", "max_depth", "n_bits", "n_features", "training_time", "compilation_time", "prediction_time", "accuracy"])    
    for max_depth, n_bits, n_features in product(max_depths, n_bits, n_features):
        X, y = make_classification(random_state=42, n_features=n_features, n_samples=1_000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        if modelname == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(max_depth=max_depth)
        elif modelname == "RandomForestClassifier":
            model = RandomForestClassifier(n_estimators=10, max_depth=max_depth)
        elif modelname == "FHEDecisionTreeClassifier":
            model = FHEDecisionTreeClassifier(max_depth=max_depth, n_bits=n_bits)
        elif modelname == "FHERandomForestClassifier":
            model = FHERandomForestClassifier(n_estimators=10, max_depth=max_depth, n_bits=n_bits)
        tic = time.perf_counter()
        model.fit(X,y)
        toc = time.perf_counter()
        train_time = toc - tic
        print(f"Training time: {train_time}")

        if modelname.startswith("FHE"):
            tic = time.perf_counter()
            model.compile(X_train)
            toc = time.perf_counter()
            compilation_time = toc - tic
            print(f"Compilation time: {compilation_time}")
        else:
            compilation_time = 0

        tic = time.perf_counter()
        if modelname.startswith("FHE"):
            y_pred = model.predict(X_test, fhe="execute")
        else:
            y_pred = model.predict(X_test)
        toc = time.perf_counter()
        prediction_time = toc - tic
        print(f"Prediction time: {prediction_time}")

        #with open(f"{MODELS_DIR}/{model}_{max_depth}_{n_bits}_{n_features}.pkl", "wb") as f:
        #    pickle.dump(model, f) 
        print(f"model name: {model}, max_depth: {max_depth}, n_bits: {n_bits}, n_features: {n_features}")
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Accuracy: {acc}, F1: {f1}")
        new_row = {"model": modelname, "max_depth": max_depth, "n_bits": n_bits, 
                    "n_features": n_features, 
                    "training_time": train_time, 
                "compilation_time": compilation_time, 
                "prediction_time": prediction_time, 
                "accuracy": acc, "f1": f1}
        stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
        stats.to_csv(f"stats_{modelname}.csv")


if __name__ == '__main__':
    with Pool(4) as p:
        print(p.map(experiment, ["DecisionTreeClassifier", "RandomForestClassifier", "FHEDecisionTreeClassifier", "FHERandomForestClassifier"]))