# simple demo script
import argparse
from sklearn.model_selection import train_test_split
from src.data.loader import load_from_zip
from src.models.cnn import HAVE_TF, prep_arrays, build_small_cnn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not HAVE_TF:
        raise SystemExit("Install tensorflow-cpu to run this script.")

    with open(args.zip, "rb") as f:
        X, y, classes, _ = load_from_zip(f.read(), args.root)

    Xtr_list, Xte_list, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
    Xtr, ytr_int, le = prep_arrays(Xtr_list, ytr, args.side)
    Xte, yte_int, _ = prep_arrays(Xte_list, yte, args.side)

    model = build_small_cnn((args.side, args.side, 1), n_classes=len(classes), lr=args.lr)
    model.fit(Xtr, ytr_int, validation_data=(Xte, yte_int), epochs=args.epochs, batch_size=args.batch, verbose=1)

if __name__ == "__main__":
    main()
