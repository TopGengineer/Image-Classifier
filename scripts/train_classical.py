# quick example, not exhaustive
import argparse, joblib
from sklearn.model_selection import train_test_split
from src.data.loader import load_from_zip
from src.features.raw import RawPixelsFeatures
from src.features.dct import TextureFeaturesCompact
from src.models.classical import make_classifier, make_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--mode", choices=["raw","texture"], default="texture")
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--clf", choices=["svm","rf","lr"], default="svm")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    with open(args.zip, "rb") as f:
        X, y, classes, _ = load_from_zip(f.read(), args.root)

    feat = TextureFeaturesCompact(args.side, args.k) if args.mode=="texture" else RawPixelsFeatures(args.side)
    if args.clf=="svm": clf = make_classifier("SVM (RBF)", {"C":1.0})
    elif args.clf=="rf": clf = make_classifier("RandomForest", {"n_estimators":300})
    else: clf = make_classifier("LogisticRegression", {"C":1.0})

    pipe = make_pipeline(feat, scale=True, use_pca=False, pca_k=64, clf=clf, seed=args.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    pipe.fit(Xtr, ytr)
    joblib.dump({"pipe":pipe, "classes":classes}, args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
