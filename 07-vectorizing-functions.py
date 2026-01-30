# BROADER CONTEXT:
# Vectorizing is a feature engineering operation - creating new columns
# as functions of existing columns. The goal is to do this efficiently
# so feature engineering is fast even on large datasets.
#
# TLDR SUMMARY:
# apply = bad
# np.vectorize = bad
# njit (for loops) = good
# numpy (for arithmetic) = good
# everything else = cant be sped up


import numpy as np
import pandas as pd
from numba import njit

# ============================================================
# PART 1: arithmetic functions can be vectorized
# ============================================================

def arith_func(x, y):
    return x + 2 * y

# ---- BAD: pandas apply (Python per-row calls) ----
def vectorized_w_apply(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        f1=df.apply(lambda r: arith_func(r["colA"], r["colB"]), axis=1)
    )


# ---- BAD: np.vectorize (still Python per-element) ----
def vectorized_w_npvectorize(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        f1=np.vectorize(lambda x, y: arith_func(x, y))(
            df["colA"].to_numpy(), df["colB"].to_numpy()
        )
    )

# ---- GOOD: true vectorization (NumPy arithmetic) ----
def vectorized_w_numpy(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(f1=df["colA"] + 2 * df["colB"])

# ============================================================
# PART 2: loop-based function
# ============================================================

def resetting_cumsum(arr, reset_threshold=0.3):
    """
    This represents a function which requires stateful looping.
    Cumulative sum that resets to 0 whenever a value falls below threshold.
    Example: [0.5, 0.6, 0.2, 0.4, 0.5] with threshold 0.3
    Result:  [0.5, 1.1, 0.0, 0.4, 0.9]
    """
    out = []
    total = 0.0
    for x in arr:
        if x < reset_threshold:
            total = 0.0
        else:
            total += x
        out.append(total)
    return np.array(out)

# ---- BAD: pandas apply (still Python loop, no speedup) ----
# apply doesn't help because the function itself is a Python loop
def vectorized_w_apply(df: pd.DataFrame) -> pd.DataFrame:
    arr = df["some_col"].to_numpy()
    result = resetting_cumsum(arr, reset_threshold=0.3)
    return df.assign(f2=result)

# ---- BAD: np.vectorize (still Python loop, no speedup) ----
# np.vectorize doesn't actually vectorize, it's just a for loop wrapper
def vectorized_w_npvectorize(df: pd.DataFrame) -> pd.DataFrame:
    arr = df["some_col"].to_numpy()
    result = resetting_cumsum(arr, reset_threshold=0.3)
    return df.assign(f2=result)

# ---- GOOD: numba njit (compiled loop) ----
# The original function needs to be rewritten with arrays instead of lists
@njit
def resetting_cumsum2(arr, reset_threshold=0.3):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64) 
    total = 0.0
    for i in range(n):
        if arr[i] < reset_threshold:
            total = 0.0
        else:
            total += arr[i]
        out[i] = total  
    return out

def vectorized_w_numba(df: pd.DataFrame) -> pd.DataFrame:
    arr = df["some_col"].to_numpy(dtype=np.float64)
    return df.assign(f2=resetting_cumsum2(arr))

# ============================================================
# PART 3: function that cannot be sped up
# string + Python objects + dynamic logic
# ============================================================

POSITIVE = {"great", "amazing"}
NEGATIVE = {"bad", "terrible"}

def complex_python_func(s: str) -> int:
    score = 0
    for token in s.lower().split():
        if token in POSITIVE:
            score += 1
        elif token in NEGATIVE:
            score -= 1
    return score


# ---- Only possible implementation: Python apply ----
def vectorized_w_apply(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        f3=df["text"].apply(complex_python_func)
    )

# ============================================================
# USAGE FOR ALL 3
# ============================================================

df_final = (
    df
    .pipe(vectorized_w_numpy)
    .pipe(vectorized_w_numba)
    .pipe(vectorized_w_apply)
)

