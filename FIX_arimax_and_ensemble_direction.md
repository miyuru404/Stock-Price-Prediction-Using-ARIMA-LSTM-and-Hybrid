# Fix: ARIMAX off-by-one + add directional accuracy to ensembles

Two changes. The first is a genuine bug that invalidates every ARIMAX number
currently in `results/mh_models/`. The second adds the missing metric.

---

## BUG 1 — ARIMAX forecasts are shifted one step early

Affects `src/mh_multimodel.py` and `src/mh_ensemble.py`.

### The problem

```python
r_o = ar.apply(lp[:o], exog=ex[:o])      # conditions on indices 0 .. o-1
fex = np.tile(ex[o-1], (max(H), 1))
fc  = r_o.get_forecast(max(H), exog=fex).predicted_mean
base = data["price"].iloc[o]
P["ARIMAX"][k] = np.array([fc[h-1]/base - 1 for h in H])
```

`lp[:o]` ends at index `o-1`. So `get_forecast(1)` returns the forecast **for
index o**, meaning `fc[0]` predicts index `o`, `fc[1]` predicts `o+1`, and in
general `fc[h-1]` predicts index `o+h-1`.

But the target being compared against is:

```python
act_ret = data[f"y{h}"].iloc[o]           # return from price[o] to price[o+h]
```

So the actual return runs `o → o+h`, while the prediction runs
`o-1 → o+h-1`. **Every ARIMAX horizon is evaluated one step early.**

At h=1 the model is effectively predicting the return that already happened,
then being scored against the next one. That explains ASPI scoring 25.35%:
below chance means systematically inverted, which is the signature of a shift,
not of a weak model.

### The fix

Condition on data **through** index `o` so the first forecast step lands on
`o+1`.

**Before**
```python
r_o = ar.apply(lp[:o], exog=ex[:o])
fex = np.tile(ex[o-1], (max(H), 1))
fc  = np.asarray(r_o.get_forecast(max(H), exog=fex).predicted_mean)
base = data["price"].iloc[o]
for h in H: ax_pred[h].append(np.exp(fc[h-1])/base - 1)
```

**After**
```python
r_o = ar.apply(lp[:o+1], exog=ex[:o+1])          # include index o
fex = np.tile(ex[o], (max(H), 1))                # hold macro at last KNOWN row
fc  = np.asarray(r_o.get_forecast(max(H), exog=fex).predicted_mean)
base = data["price"].iloc[o]                     # now the last observed price
for h in H: ax_pred[h].append(np.exp(fc[h-1])/base - 1)
```

Same change in `mh_ensemble.py` inside `preds_at()`:

**Before**
```python
r_o = ar.apply(lp[:o], exog=ex[:o]); fex = np.tile(ex[o-1], (max(H), 1))
...
base = data["price"].iloc[o]; P["ARIMAX"][k] = np.array([f[h-1]/base-1 for h in H])
```

**After**
```python
r_o = ar.apply(lp[:o+1], exog=ex[:o+1]); fex = np.tile(ex[o], (max(H), 1))
...
base = data["price"].iloc[o]; P["ARIMAX"][k] = np.array([f[h-1]/base-1 for h in H])
```

### Guard against a repeat

Add an assertion after building predictions:

```python
# ARIMAX at h=1 should not be systematically inverted.
# Below ~40% on a large test set indicates an alignment error, not a weak model.
d1 = metrics(np.array(ax_pred[1]), 1, ax_oidx)[0]
assert d1 > 40, f"ARIMAX h=1 direction {d1:.1f}% - check forecast alignment"
```

### Also verify

Confirm the last exogenous row used is the last **known** one, not a future
value. After the fix `ex[o]` is observed at time `o`, so holding it constant
across the forecast horizon is correct and leak-free.

---

## BUG 2 — "Naive" row mixes two different baselines

In `mh_multimodel.py`:

```python
pers = np.zeros(len(test_idx))                    # persistence: pred return = 0
dir_maj = max(np.mean(act>0), np.mean(act<0))*100 # majority class
_, mape = metrics(pers, h, test_idx)
rows.append({"model":"Naive","horizon":h,"dir_acc":dir_maj,"price_mape":mape})
```

The `dir_acc` is majority-class; the `price_mape` is persistence. Two different
models on one row. Split them:

```python
for h in H:
    act  = data[f"y{h}"].iloc[test_idx].values
    m    = np.abs(act) > 1e-9                     # same flat-day mask as metrics()
    pers = np.zeros(len(test_idx))
    _, mape_pers = metrics(pers, h, test_idx)

    # Majority-class direction baseline (no price forecast -> MAPE not defined)
    dir_maj = max(np.mean(act[m] > 0), np.mean(act[m] < 0)) * 100
    rows.append({"model":"Majority","horizon":h,
                 "dir_acc":dir_maj, "price_mape":np.nan})

    # Persistence baseline: predicted return = 0 -> direction is a tie, count as 0
    dir_pers = 0.0
    rows.append({"model":"Persistence","horizon":h,
                 "dir_acc":dir_pers, "price_mape":mape_pers})
```

Note `sign(0) == 0` never matches `sign(act) == +/-1`, so a zero-return
prediction scores 0% direction by construction. If a persistence *direction*
baseline is wanted, use the sign of the previous realised return instead:

```python
prev_ret = data["y1"].shift(1).iloc[test_idx].values
dir_pers = np.mean(np.sign(prev_ret[m]) == np.sign(act[m])) * 100
```

Update the `reindex([...])` lists and the plotting loop to the new model names.

---

## ADDITION — directional accuracy for the ensembles

`src/mh_ensemble.py` currently reports MAPE only, so the central question
("does averaging ARIMAX and RandomForest beat either on *direction*?") cannot
be answered from the output.

### Add alongside `mape_per_h`

```python
def dir_per_h(pred, origins):
    """Directional accuracy per horizon for a (n_origins, n_horizons) array."""
    out = []
    oi = np.array(origins)
    for j, h in enumerate(H):
        act = data[f"y{h}"].iloc[oi].values
        m = np.abs(act) > 1e-9                      # drop flat outcomes
        out.append(np.mean(np.sign(pred[m, j]) == np.sign(act[m])) * 100)
    return out
```

### Add a majority-class reference row

```python
def dir_majority(origins):
    out = []
    oi = np.array(origins)
    for h in H:
        act = data[f"y{h}"].iloc[oi].values
        m = np.abs(act) > 1e-9
        out.append(max(np.mean(act[m] > 0), np.mean(act[m] < 0)) * 100)
    return out
```

### Emit both tables

```python
tab_mape = pd.DataFrame({m: mape_per_h(M[m], test_origins) for m in M},
                        index=HLAB).T
tab_dir  = pd.DataFrame({m: dir_per_h(M[m],  test_origins) for m in M},
                        index=HLAB).T
tab_dir.loc["Majority"] = dir_majority(test_origins)     # reference row

print("\n=== MAPE (%) ===");               print(tab_mape.round(3).to_string())
print("\n=== Directional accuracy (%) ==="); print(tab_dir.round(1).to_string())

tab_mape.round(3).to_csv(OUT / f"{T}_ensemble_mape.csv")
tab_dir.round(2).to_csv(OUT / f"{T}_ensemble_direction.csv")
```

### Important — SWITCH must be selected on direction, not MAPE

`switch_pick` currently chooses the per-horizon winner by validation **MAPE**.
If SWITCH is to be judged on direction, it must also be *selected* on
direction, or the comparison is inconsistent.

```python
val_dir = {m: dir_per_h(Pval[m], val_origins) for m in Pval}
switch_pick_dir = [max(["ARIMAX", "RandomForest", "CNN-LSTM"],
                       key=lambda m: val_dir[m][j]) for j in range(len(H))]
```

Report `SWITCH-mape` and `SWITCH-dir` as separate rows. Selecting on the test
set would be leakage — keep both picks on the validation slice.

---

## After running

1. Confirm ARIMAX h=1 is no longer below 40% anywhere.
2. Compare new ARIMAX numbers against the old ones in
   `results/mh_models/*_results.csv` and record the difference — the old values
   should be treated as void.
3. Note in the write-up that long horizons (h >= 44) are dominated by upward
   drift: the majority-class baseline rises with h, so high accuracy there
   reflects trend, not skill. Report h = 1, 5, 22 as the meaningful set.
4. Re-run every ticker so the results are internally consistent.
