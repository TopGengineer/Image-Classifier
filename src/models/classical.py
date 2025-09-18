from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def make_classifier(name: str, params: Dict):
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", C=params.get("C", 1.0), gamma="scale", probability=True)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 300), random_state=42)
    if name == "LogisticRegression":
        return LogisticRegression(C=params.get("C", 1.0), max_iter=500)
    raise ValueError("Unknown classifier.")

def make_pipeline(feat_transformer, scale=True, use_pca=False, pca_k=64, clf=None, seed=123):
    steps = [("feat", feat_transformer)]
    if scale:
        steps.append(("scale", StandardScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_k, whiten=True, random_state=seed)))
    steps.append(("clf", clf))
    return Pipeline(steps)
