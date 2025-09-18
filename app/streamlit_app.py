import io, zipfile
import numpy as np
import streamlit as st

from src.data.loader import load_from_zip, list_roots_in_zip
from src.features.raw import RawPixelsFeatures
from src.features.dct import TextureFeaturesCompact
from src.models.classical import make_classifier, make_pipeline
from src.utils.eval import evaluate_classifier
from src.utils.plotting import plot_confusion_matrix

# optional CNN
try:
    from src.models.cnn import HAVE_TF, prep_arrays, build_small_cnn
except Exception:
    HAVE_TF = False

st.set_page_config(page_title="Image Classification — Prototype", layout="wide")
st.title("Image Classification — Prototype")
st.caption("Runs locally during this session. No data is stored.")
st.markdown("---")

def ok(m): st.success(f"✅ {m}")
def warn(m): st.warning(f"⚠️ {m}")
def err(m): st.error(f"❌ {m}")

# Session
ss = st.session_state
for k, v in dict(X=None, y=None, classes=None,
                 feat_cfg=None, feat_ready=False,
                 pipeline=None, Xtr=None, Xte=None, ytr=None, yte=None,
                 cnn_model=None, cnn_side=None, le=None).items():
    if k not in ss: ss[k] = v

# Step 1 — Load
st.markdown("### Step 1 · Load images")
col1, col2 = st.columns([2,1])
with col1:
    upzip = st.file_uploader("Upload a ZIP structured as <ROOT>/<class>/* (nested roots OK)", type=["zip"])
with col2:
    limit = st.slider("(Optional) limit per class", 0, 5000, 0, help="0 = no limit")

chosen_root = None
if upzip:
    try:
        roots = list_roots_in_zip(upzip.getvalue())
        if not roots:
            err("Could not detect `<ROOT>/<class>/*`. Check folder layout.")
        else:
            options = sorted(roots.keys(), key=lambda r: (-len(roots[r]), -sum(roots[r].values())))
            def _fmt(r): return f"{r} — {len(roots[r])} classes, {sum(roots[r].values())} images"
            chosen_root = st.selectbox("Pick dataset ROOT inside the ZIP", options, format_func=_fmt)
    except Exception as e:
        err(f"ZIP scan failed: {e}")

if st.button("Load ZIP", type="primary", disabled=(upzip is None)):
    try:
        X, y, classes, report = load_from_zip(
            upzip.getvalue(),
            chosen_root,
            (None if limit==0 else limit)
        )
        ss.X, ss.y, ss.classes = X, y, classes
        ok(f"Loaded {len(y)} images · Classes: {classes}")
        st.write("Per-class counts:", report["class_counts"])
        if report.get("skipped", 0):
            warn(f"Skipped {report['skipped']} files (unsupported/corrupt).")
        cols = st.columns(min(6, len(X)))
        for i in range(min(6, len(X))):
            cols[i].image(X[i], caption=str(y[i]), use_container_width=True)
    except Exception as e:
        err(f"Load failed: {e}")

st.markdown("---")

# Step 2 — Features
st.markdown("### Step 2 · Features")
use_texture = st.toggle("Use compact texture features (classical ML)", value=False,
                        help="OFF = raw pixels (simplest). ON = compact texture features (DCT).")
colL, colR = st.columns(2)
with colL:
    side = st.slider("Resize side (pixels)", 32, 256, 64, step=8,
                     help="Images are resized to side×side before learning.")
with colR:
    if use_texture:
        kblk = st.slider("Feature detail (k×k block)", 4, min(96, side), 16, step=2,
                         help="Higher keeps more detail; lower is faster.")
        scale = st.checkbox("Standardize features", True)
        use_pca = st.checkbox("Use PCA (optional)", False)
        pca_k = st.slider("PCA components", 2, 256, 64, step=2, disabled=not use_pca)
    else:
        kblk = None
        scale = st.checkbox("Standardize features", True)
        use_pca = False
        pca_k = None

if st.button("Apply features", type="primary"):
    ss.feat_cfg = dict(mode=("texture" if use_texture else "raw"),
                       side=side, k=kblk, scale=scale, use_pca=use_pca, pca_k=pca_k)
    ss.feat_ready = True
    ok(("Texture (DCT)" if use_texture else "Raw") + f" · resize={side}px")

st.markdown("---")

# Step 3 — Train
st.markdown("### Step 3 · Train")
ready_data = (ss.X is not None and ss.y is not None and ss.classes is not None and len(set(ss.y))>=2)
ready_feats = ss.feat_ready and ss.feat_cfg is not None

train_mode = st.radio("Training approach",
                      ["Classical ML (SVM / RandomForest / LogisticRegression)", "Small CNN (Keras)"],
                      horizontal=False)

cA, cB, cC = st.columns([1,1,1])
with cB:
    test_size = st.slider("Test size (kept aside)", 0.1, 0.4, 0.2, step=0.05)
with cC:
    seed = st.number_input("Seed (repeatable split)", 0, 9999, 123, step=1)

from sklearn.model_selection import train_test_split

if train_mode.startswith("Classical"):
    with cA:
        clf_name = st.selectbox("Classifier", ["SVM (RBF)","RandomForest","LogisticRegression"])
    # knobs
    if clf_name == "SVM (RBF)":
        C = st.slider("Model flexibility (C)", 0.01, 10.0, 1.0, 0.01)
        mparams = dict(C=C)
    elif clf_name == "RandomForest":
        n_estimators = st.slider("Number of trees", 50, 800, 300, step=50)
        mparams = dict(n_estimators=n_estimators)
    else:
        C = st.slider("Model flexibility (C)", 0.01, 10.0, 1.0, 0.01)
        mparams = dict(C=C)

    if st.button("Train model", type="primary", disabled=not(ready_data and ready_feats)):
        try:
            Xtr, Xte, ytr, yte = train_test_split(ss.X, ss.y, test_size=test_size, stratify=ss.y, random_state=seed)
        except ValueError:
            warn("Some classes are very small — using random split (not stratified).")
            Xtr, Xte, ytr, yte = train_test_split(ss.X, ss.y, test_size=test_size, shuffle=True, random_state=seed)

        # build pipeline
        if ss.feat_cfg["mode"] == "texture":
            feat = TextureFeaturesCompact(side=ss.feat_cfg["side"], k=ss.feat_cfg["k"])
        else:
            feat = RawPixelsFeatures(side=ss.feat_cfg["side"])
        clf = make_classifier(clf_name, mparams)
        pipe = make_pipeline(feat, scale=ss.feat_cfg["scale"],
                             use_pca=ss.feat_cfg["use_pca"], pca_k=ss.feat_cfg.get("pca_k", 64),
                             clf=clf, seed=seed)
        pipe.fit(Xtr, ytr)
        ss.pipeline = pipe
        ss.cnn_model = None
        ss.Xtr, ss.Xte, ss.ytr, ss.yte = Xtr, Xte, ytr, yte
        ok(f"Training complete. Train={len(ytr)} · Test={len(yte)}")

else:
    if not HAVE_TF:
        warn("TensorFlow not installed. Install `tensorflow-cpu` to enable CNN.")
    epochs = st.slider("Epochs", 2, 50, 8, step=1)
    batch_size = st.slider("Batch size", 8, 128, 32, step=8)
    lr = float(st.selectbox("Learning rate", ["1e-3","5e-4","1e-4"], index=0))

    if st.button("Train CNN", type="primary", disabled=not(ready_data and HAVE_TF)):
        try:
            Xtr_list, Xte_list, ytr, yte = train_test_split(ss.X, ss.y, test_size=test_size, stratify=ss.y, random_state=seed)
        except ValueError:
            warn("Some classes are very small — using random split (not stratified).")
            Xtr_list, Xte_list, ytr, yte = train_test_split(ss.X, ss.y, test_size=test_size, shuffle=True, random_state=seed)

        side = ss.feat_cfg["side"] if ss.feat_cfg else 64
        Xtr, ytr_int, le = prep_arrays(Xtr_list, ytr, side)
        Xte, yte_int, _ = prep_arrays(Xte_list, yte, side)
        model = build_small_cnn((side, side, 1), n_classes=len(ss.classes), lr=lr)
        model.fit(Xtr, ytr_int, validation_data=(Xte, yte_int), epochs=epochs, batch_size=batch_size, verbose=0)

        ss.cnn_model = model
        ss.cnn_side = side
        ss.le = le
        ss.pipeline = None
        ss.Xtr, ss.Xte, ss.ytr, ss.yte = Xtr_list, Xte_list, ytr, yte
        ok(f"CNN training complete. Train={len(ytr)} · Test={len(yte)}")

st.markdown("---")

# Step 4 — Evaluate & Predict
st.markdown("### Step 4 · Evaluate & Predict")
cE, cP = st.columns([1,1])

with cE:
    if st.button("Evaluate on TEST"):
        if ss.pipeline is not None and ss.Xte is not None:
            acc, cm, report = evaluate_classifier(ss.pipeline, ss.Xte, ss.yte, ss.classes)
            ok(f"Test Accuracy: **{acc:.3f}**")
            fig = plot_confusion_matrix(cm, ss.classes)
            st.pyplot(fig)
            st.text("Classification report")
            st.code(report, language="text")
        elif ss.cnn_model is not None and ss.Xte is not None and HAVE_TF:
            from sklearn.preprocessing import LabelEncoder
            # evaluate
            side = ss.cnn_side or 64
            X = np.zeros((len(ss.Xte), side, side, 1), dtype="float32")
            for i, im in enumerate(ss.Xte):
                g = np.asarray(im.convert("L").resize((side, side)), dtype=np.float32) / 255.0
                X[i, :, :, 0] = g
            y_int = ss.le.transform(ss.yte)
            loss, acc = ss.cnn_model.evaluate(X, y_int, verbose=0)
            ok(f"Test Accuracy: **{acc:.3f}**")
            yprob = ss.cnn_model.predict(X, verbose=0)
            yhat_int = np.argmax(yprob, axis=1)
            yhat = ss.le.inverse_transform(yhat_int)
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(ss.yte, yhat, labels=ss.classes)
            fig = plot_confusion_matrix(cm, ss.classes)
            st.pyplot(fig)
            st.text("Classification report")
            st.code(classification_report(ss.yte, yhat, labels=ss.classes), language="text")
        else:
            err("Train a model first.")

with cP:
    st.subheader("Predict on a single image")
    up = st.file_uploader("Upload ONE image", type=["png","jpg","jpeg","bmp","tif","tiff","dcm"], key="predict_img")
    if up:
        raw = up.read()
        from src.data.loader import _decode_image
        img = _decode_image(up.name, raw)
        if img is None:
            err("Unsupported or corrupt image (or install pydicom for .dcm).")
        else:
            st.image(img, caption="Input", use_container_width=True)
            if ss.pipeline is not None:
                pred = ss.pipeline.predict([img])[0]
                st.write("**Prediction:**", str(pred))
                if hasattr(ss.pipeline, "predict_proba"):
                    proba = ss.pipeline.predict_proba([img])[0]
                    st.write({cls: float(proba[i]) for i, cls in enumerate(ss.classes)})
            elif ss.cnn_model is not None and HAVE_TF:
                side = ss.cnn_side or 64
                g = np.asarray(img.convert("L").resize((side, side)), dtype=np.float32) / 255.0
                x = g.reshape(1, side, side, 1)
                yprob = ss.cnn_model.predict(x, verbose=0)[0]
                idx = int(np.argmax(yprob))
                pred = ss.le.inverse_transform([idx])[0]
                st.write("**Prediction:**", str(pred))
                st.write({cls: float(yprob[i]) for i, cls in enumerate(ss.classes)})
            else:
                warn("Train a model first.")
