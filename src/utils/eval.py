from typing import Sequence
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_classifier(clf, Xte, yte, class_names: Sequence[str]):
    yhat = clf.predict(Xte)
    acc = accuracy_score(yte, yhat)
    cm = confusion_matrix(yte, yhat, labels=class_names)
    report = classification_report(yte, yhat, labels=class_names)
    return acc, cm, report
