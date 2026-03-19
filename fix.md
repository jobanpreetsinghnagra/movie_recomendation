Your build is **successful**, but your app is **crashing at runtime due to memory limits (512MB)** — not a code error.

### 🔴 Root Problem

```
==> Out of memory (used over 512Mi)
```

This happens during:

```
Loading model checkpoint from models/model_weights.pth
Loading ratings.parquet
```

👉 So your ML model + data is **too large for Render’s free tier RAM**

---

# ✅ Fix Options (choose one)

## 🟢 Option 1: Reduce Memory Usage (Best for free tier)

### 1. Don’t load everything at startup

Right now you're doing:

```python
load model
load movies.csv
load ratings.parquet
```

👉 This loads everything into RAM immediately.

### Fix:

Load lazily (only when needed):

```python
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model
```

---

### 2. Remove `ratings.parquet` loading

This line is killing you:

```
Loading ratings.parquet to reconstruct mappings
```

👉 Parquet + pandas = heavy memory usage

### Fix:

* Precompute mappings offline
* Save as small JSON or pickle

Example:

```python
# offline script
save mappings.json
```

Then in API:

```python
with open("mappings.json") as f:
    mappings = json.load(f)
```

---

### 3. Use smaller model

Your `.pth` file is likely big.

👉 Try:

* Reduce embedding size
* Quantize model
* Use CPU-friendly version

---

### 4. Avoid pandas if possible

Instead of:

```python
pd.read_parquet()
```

Use:

```python
pyarrow.dataset (more efficient)
```

or convert to:

* CSV (chunked read)
* JSON (small)

---

## 🟡 Option 2: Upgrade Render Plan (Quick Fix)

If you don’t want to optimize:

👉 Upgrade to:

* **Starter plan (1GB RAM)** or higher

This will likely fix instantly.

---

## 🔵 Option 3: Move Model Off Backend (Best Architecture)

Instead of loading model in FastAPI:

👉 Use:

* separate ML service
* or precomputed recommendations

Example:

```python
GET /recommend?user_id=123
```

Returns from precomputed DB (no heavy model in RAM)

---

## ⚠️ Also Fix This (Important)

You see:

```
No open ports detected
```

Even though uvicorn runs:

```bash
uvicorn api:app --host 0.0.0.0 --port 5000
```

👉 Render expects:

```bash
--port $PORT
```

### Fix:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

---

# 🚀 Recommended Combo (best for you)

Since you're building an ML app:

1. Precompute mappings ✅
2. Remove `ratings.parquet` loading ❌
3. Lazy load model ✅
4. Use `$PORT` ✅

---

# 💬 If you want

Send me:

* your `api.py`
* model size (`.pth`)
* dataset size

I can **optimize it to run under 512MB** (production-ready).
