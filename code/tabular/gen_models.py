
MODELS_DIR = "./pickle_field"
from sklearn.metrics import accuracy_score, f1_score
from concrete.ml.sklearn import DecisionTreeClassifier as FHEDecisionTreeClassifier, RandomForestClassifier as FHERandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from sklearn.model_selection import train_test_split
import pickle
import time
import pandas as pd
import numpy as np

df = pd.read_csv('data_arrhythmia.csv', delimiter = ';',na_values = ['?'])

#setting prediction values to "Normal" or "Risk" based on column scores values
df.loc[df["diagnosis"] == 1,"diagnosis"] = 0    #class 1 is normal arrythmia
df.loc[df["diagnosis"] != 0,"diagnosis"] = 1    #other classes are risk classes
print(df.head())

X_df = df.loc[:,df.columns != 'diagnosis'] #select all features except target feature
X = np.array(X_df) #convert it to array (Simple Imputer doesn't work with dataframes)

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='median') #create imp object to impute median in all missing values 
imp = imp.fit(X) #calculate median values of the features with missing values
X = imp.transform(X) #fill dataset with median values wherever finds missing values
y = df.iloc[:,-1] #subset target label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

# models = ["DecisionTreeClassifier"]# , "RandomForestClassifier"]
# max_depths = range(3, 6)
# n_bits = range(2, 4)
# n_features = range(5, 8)

models = ["DecisionTreeClassifier", "RandomForestClassifier", "FHEDecisionTreeClassifier", "FHERandomForestClassifier"]
max_depths = range(3, 15)
n_bits = range(2, 12)
 
stats = pd.DataFrame(columns=["model", "max_depth", "n_bits", "training_time", "compilation_time", "prediction_time", "accuracy"])    
for modelname, max_depth, n_bits in product(models, max_depths, n_bits):
    # X, y = make_classification(random_state=42, n_features=n_features, n_samples=1_000)
    if modelname == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
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

        try:
            model.compile(X_train)
            toc = time.perf_counter()
            compilation_time = toc - tic
            print(f"Compilation time: {compilation_time}")
        except Exception as e:
            print(f"Error al compilar {modelname} con max_depth={max_depth}, n_bits={n_bits}: {e}")
            new_row = {"model": modelname, "max_depth": max_depth, "n_bits": n_bits, 
               "training_time": 0, 
               "compilation_time": 0, 
               "prediction_time": 0, 
                "accuracy": 0, "f1": 0}
            stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)

            continue
        #continue  # Saltar esta iteración
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
    print(f"model name: {model}, max_depth: {max_depth}, n_bits: {n_bits}")
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    new_row = {"model": modelname, "max_depth": max_depth, "n_bits": n_bits, 
               "training_time": train_time, 
               "compilation_time": compilation_time, 
               "prediction_time": prediction_time, 
                "accuracy": acc, "f1": f1}
    stats = pd.concat([stats, pd.DataFrame([new_row])], ignore_index=True)
    stats.to_csv(f"stats_{modelname}.csv")