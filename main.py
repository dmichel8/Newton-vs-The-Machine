import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X = pd.read_csv("train_X.csv", header=None)
Y = pd.read_csv("train_Y.csv", header=None)

X_df = X.copy()
Y_df = Y.copy()

# Some online references said to scale, though I do not know if its needed or not. Forgive me for not knowing that.
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# 1% test
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.01, random_state=8)

model = Sequential()
for i in range(10):
    model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='linear'))

optimizer = Adam()
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Paper specifies 5000 batch size
history = model.fit(
    X_train, Y_train,
    epochs=1000,
    batch_size=5000,
    validation_data=(X_val, Y_val)
)

test_loss = model.evaluate(X_val, Y_val)
print(f"Final Test MAE: {test_loss:.5f}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training MAE')
plt.plot(history.history['val_loss'], label='Validation MAE', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Mean Absolute Error vs. Epoch')
plt.legend()
plt.grid(True)
plt.savefig('mae_vs_epoch.png')

def _inverse_or_identity(arr, scaler):
    if scaler is None:
        return arr
    # If the scaler was fit on a DataFrame, ensure shapes match
    return scaler.inverse_transform(arr)


def plot_trajectories_grouped(X_like, Y_like, model,
                              scaler_X=None, scaler_Y=None,
                              out_png="trajectories_example.png",
                              t_max=None, pred_batch_size=65536, round_dec=6):
    X = np.asarray(X_like); Y = np.asarray(Y_like)
    if X.ndim != 2 or X.shape[1] != 3: raise ValueError(f"X must be (N,3), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 4: raise ValueError(f"Y must be (N,4), got {Y.shape}")

    keys = np.round(X[:,1:3], round_dec)
    uniq, idx, inv, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    grp = np.argmax(counts)                  # largest group = most time points
    pair = uniq[grp]                         # chosen initial x2
    mask = (np.isclose(X[:,1], pair[0]) & np.isclose(X[:,2], pair[1]))
    Xs, Ys = X[mask], Y[mask]

    ordr = np.argsort(Xs[:,0])               # sort by time
    Xs, Ys = Xs[ordr], Ys[ordr]

    if t_max is not None:
        m = Xs[:,0] <= float(t_max)
        Xs, Ys = Xs[m], Ys[m]

    x2x0, x2y0 = Xs[0,1], Xs[0,2]
    t = Xs[:,0]
    Xin = np.column_stack([t, np.full_like(t, x2x0), np.full_like(t, x2y0)])
    if scaler_X is not None: Xin = scaler_X.transform(Xin)

    Yp_s = model.predict(Xin, verbose=0, batch_size=pred_batch_size)
    Yp = Yp_s if scaler_Y is None else scaler_Y.inverse_transform(Yp_s)

    Yt = Ys
    try:
        if scaler_Y is not None and (np.nanmean(np.abs(Yt)) < 0.5 and np.nanstd(Yt) < 1.5):
            Yt = scaler_Y.inverse_transform(Yt)
    except Exception:
        pass

    x1x_t, x1y_t, x2x_t, x2y_t = Yt.T
    x1x_p, x1y_p, x2x_p, x2y_p = Yp.T
    x3x_t, x3y_t = -(x1x_t + x2x_t), -(x1y_t + x2y_t)
    x3x_p, x3y_p = -(x1x_p + x2x_p), -(x1y_p + x2y_p)

    plt.figure(figsize=(7.5, 7.5))
    plt.plot(x1x_t, x1y_t, '--', label='x1 (Brutus)')
    plt.plot(x2x_t, x2y_t, '--', label='x2 (Brutus)')
    plt.plot(x3x_t, x3y_t, '--', label='x3 (Brutus)')
    plt.plot(x1x_p, x1y_p, label='x1 (ANN)')
    plt.plot(x2x_p, x2y_p, label='x2 (ANN)')
    plt.plot(x3x_p, x3y_p, label='x3 (ANN)')
    plt.scatter([x1x_t[0], x2x_t[0], x3x_t[0]],[x1y_t[0], x2y_t[0], x3y_t[0]], s=30, zorder=5)
    plt.title('Three-body trajectories: ANN vs Brutus')
    plt.xlabel('x'); plt.ylabel('y'); plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()
    print(f"Saved {out_png} | points: {len(t)} | t:[{t.min():.3g},{t.max():.3g}] | group size:{counts[grp]} | pair:{pair}")


def plot_trajectories_fast(X_df, Y_df, model,
                           scaler_X=None, scaler_Y=None,
                           out_png="trajectories_example.png",
                           t_max=None, pred_batch_size=65536):
    import numpy as np, matplotlib.pyplot as plt

    X_arr = np.asarray(X_df)
    Y_arr = np.asarray(Y_df)
    if X_arr.shape[1] != 3 or Y_arr.shape[1] != 4:
        raise ValueError(f"Expected X(N,3) and Y(N,4); got {X_arr.shape}, {Y_arr.shape}")

    # one contiguous trajectory (find where time decreases = new sim)
    t_all = X_arr[:,0]
    cut = np.argmax(np.diff(t_all) < 0) + 1
    if cut == 1 and not (np.diff(t_all) < 0).any():  # no reset found; fall back to first 2561 rows
        cut = min(len(t_all), 2561)
    X_sim = X_arr[:cut]
    Y_sim = Y_arr[:cut]

    # optional time mask
    if t_max is not None:
        m = X_sim[:,0] <= float(t_max)
        X_sim = X_sim[m]; Y_sim = Y_sim[m]

    # inputs [t, x2x0, x2y0] with constant initial x2 from first row
    x2x0, x2y0 = X_sim[0,1], X_sim[0,2]
    t = X_sim[:,0]
    X_in = np.column_stack([t, np.full_like(t, x2x0), np.full_like(t, x2y0)])
    if scaler_X is not None:
        X_in = scaler_X.transform(X_in)

    # fast predict
    Y_pred_s = model.predict(X_in, verbose=0, batch_size=pred_batch_size)
    Y_pred = Y_pred_s if scaler_Y is None else scaler_Y.inverse_transform(Y_pred_s)

    # ground truth (inverse if needed)
    Y_true = Y_sim
    try:
        if scaler_Y is not None and (np.nanmean(np.abs(Y_true)) < 0.5 and np.nanstd(Y_true) < 1.5):
            Y_true = scaler_Y.inverse_transform(Y_true)
    except Exception:
        pass

    x1x_t, x1y_t, x2x_t, x2y_t = Y_true.T
    x1x_p, x1y_p, x2x_p, x2y_p = Y_pred.T
    x3x_t, x3y_t = -(x1x_t + x2x_t), -(x1y_t + x2y_t)
    x3x_p, x3y_p = -(x1x_p + x2x_p), -(x1y_p + x2y_p)

    plt.figure(figsize=(7.5, 7.5))
    plt.plot(x1x_t, x1y_t, '--', label='x1 (Brutus)')
    plt.plot(x2x_t, x2y_t, '--', label='x2 (Brutus)')
    plt.plot(x3x_t, x3y_t, '--', label='x3 (Brutus)')
    plt.plot(x1x_p, x1y_p, label='x1 (ANN)')
    plt.plot(x2x_p, x2y_p, label='x2 (ANN)')
    plt.plot(x3x_p, x3y_p, label='x3 (ANN)')
    plt.scatter([x1x_t[0], x2x_t[0], x3x_t[0]],[x1y_t[0], x2y_t[0], x3y_t[0]], s=30, zorder=5)
    plt.title('Three-body trajectories: ANN vs Brutus')
    plt.xlabel('x'); plt.ylabel('y'); plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()
    print(f"Saved figure to {out_png} with {len(t)} points (batch_size={pred_batch_size})")

plot_trajectories_grouped(
    X_df, Y_df, model,
    scaler_X=scaler_X, scaler_Y=scaler_Y,
    out_png="trajectories_example.png",
    t_max=None,          # or 3.9 to mimic paper’s shortest window
    pred_batch_size=65536,
    round_dec=6
)


#model.save("three_body_ann_model.h5")

