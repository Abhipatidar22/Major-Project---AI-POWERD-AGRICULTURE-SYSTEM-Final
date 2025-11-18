import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def try_import_tf():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        return False

def load_keras_model(path):
    # Import lazily; caller should guard for availability.
    from tensorflow import keras
    return keras.models.load_model(path)

def topk_probs(probs, classes, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in idx]

def perm_importance_table(model, X: pd.DataFrame):
    try:
        r = permutation_importance(model, X, model.predict_proba, n_repeats=2, random_state=42)
        imp = sorted([(X.columns[i], float(m)) for i,m in enumerate(r.importances_mean)], key=lambda x: -x[1])
        return pd.DataFrame(imp, columns=['feature','importance'])
    except Exception:
        return pd.DataFrame({'feature': X.columns, 'importance': 0.0})
