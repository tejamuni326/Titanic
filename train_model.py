import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Titanic-Dataset (1).csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())


# ============================================================
# 2. FEATURE SELECTION
# ============================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ============================================================
# 5. MACHINE LEARNING MODELS
# ============================================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

decision_tree = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])

random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

gradient_boosting = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        random_state=42
    ))
])

xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])


ml_models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "XGBoost": xgb_model
}


# ============================================================
# 6. TRAIN MACHINE LEARNING MODELS
# ============================================================

ml_results = []
trained_models = {}

for name, model in ml_models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    ml_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, y_prob
        )
    })

    trained_models[name] = model


ml_results_df = pd.DataFrame(ml_results)

print("\nMachine Learning Results:")
print(ml_results_df)


# ============================================================
# 7. DEEP LEARNING PREPROCESSING
# ============================================================

dl_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

dl_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

dl_preprocessor = ColumnTransformer([
    ("num", dl_numeric, numeric_features),
    ("cat", dl_categorical, categorical_features)
])


# ============================================================
# THESE ARE THE TWO FIXED LINES
# ============================================================

X_train_dl = dl_preprocessor.fit_transform(X_train)
X_test_dl = dl_preprocessor.transform(X_test)


# ============================================================
# 8. ANN MODEL
# ============================================================

print("\nTraining: ANN")

ann_model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train_dl.shape[1],)
    ),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ann_model.fit(
    X_train_dl,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=0
)

ann_prob = ann_model.predict(
    X_test_dl,
    verbose=0
).ravel()

ann_pred = (ann_prob >= 0.5).astype(int)

ann_results = {
    "Model": "ANN",
    "Accuracy": accuracy_score(
        y_test, ann_pred
    ),
    "Precision": precision_score(
        y_test, ann_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, ann_pred, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, ann_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test, ann_prob
    )
}


# ============================================================
# 9. 1D CNN MODEL
# ============================================================

print("\nTraining: 1D CNN")

X_train_cnn = X_train_dl.reshape(
    X_train_dl.shape[0],
    X_train_dl.shape[1],
    1
)

X_test_cnn = X_test_dl.reshape(
    X_test_dl.shape[0],
    X_test_dl.shape[1],
    1
)

cnn_model = Sequential([
    Conv1D(
        32,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train_cnn.shape[1], 1)
    ),
    MaxPooling1D(pool_size=2),
    Conv1D(
        64,
        kernel_size=3,
        activation="relu"
    ),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.30),
    Dense(1, activation="sigmoid")
])

cnn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=0
)

cnn_prob = cnn_model.predict(
    X_test_cnn,
    verbose=0
).ravel()

cnn_pred = (cnn_prob >= 0.5).astype(int)

cnn_results = {
    "Model": "1D CNN",
    "Accuracy": accuracy_score(
        y_test, cnn_pred
    ),
    "Precision": precision_score(
        y_test, cnn_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, cnn_pred, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, cnn_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test, cnn_prob
    )
}


# ============================================================
# 10. COMBINE ALL MODEL RESULTS
# ============================================================

dl_results_df = pd.DataFrame([
    ann_results,
    cnn_results
])

all_results = pd.concat(
    [ml_results_df, dl_results_df],
    ignore_index=True
)

all_results = all_results.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


print("\n===================================")
print("ALL MODEL RESULTS")
print("===================================")

print(all_results)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

all_results.to_csv(
    "results/model_results.csv",
    index=False
)


# ============================================================
# 12. SELECT BEST ML MODEL
# ============================================================

best_model_name = ml_results_df.loc[
    ml_results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[
    best_model_name
]

print("\nBest ML Model:", best_model_name)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/titanic_best_model.pkl"
)

print(
    "\nSaved: models/titanic_best_model.pkl"
)


# ============================================================
# 14. SAVE TEST DATA
# ============================================================

test_data = X_test.copy()

test_data["Survived"] = y_test.values

test_data.to_csv(
    "results/test_data.csv",
    index=False
)
# ============================================================
# 16. MODEL COMPARISON GRAPH
# ============================================================

os.makedirs("results/graphs", exist_ok=True)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

plt.figure(figsize=(14, 7))

all_results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Machine Learning and Deep Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "results/graphs/model_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

best_model_pred = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_model_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.tight_layout()

plt.savefig(
    "results/graphs/confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    best_model_pred
)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(report)

with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

try:

    model_inside_pipeline = best_model.named_steps["model"]

    preprocessor_inside_pipeline = (
        best_model.named_steps["preprocessor"]
    )

    feature_names = (
        preprocessor_inside_pipeline
        .get_feature_names_out()
    )

    importance = model_inside_pipeline.feature_importances_

    feature_importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    print("\n===================================")
    print("FEATURE IMPORTANCE")
    print("===================================")

    print(feature_importance_df)


    feature_importance_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_importance_df.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Top 10 Feature Importance - "
        + best_model_name
    )

    plt.tight_layout()

    plt.savefig(
        "results/graphs/feature_importance.png",
        dpi=300
    )

    plt.close()

except Exception as e:

    print(
        "\nFeature importance could not be generated:"
    )

    print(e)


# ============================================================
# 20. SAVE MODEL RESULTS
# ============================================================

all_results.to_csv(
    "results/model_results.csv",
    index=False
)


print("\n===================================")
print("EVALUATION FILES CREATED")
print("===================================")

print(
    "results/graphs/model_comparison.png"
)

print(
    "results/graphs/confusion_matrix.png"
)

print(
    "results/graphs/feature_importance.png"
)

print(
    "results/classification_report.txt"
)

print(
    "results/feature_importance.csv"
)

# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")

print("Best ML Model:", best_model_name)

print("\nGenerated files:")
print("results/model_results.csv")
print("results/test_data.csv")
print("models/titanic_best_model.pkl")

print("\nRun Streamlit with:")
print("streamlit run app.py")