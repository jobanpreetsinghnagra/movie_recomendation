# CineMind

A movie recommender system built with **Matrix Factorization** using PyTorch's `nn.Embedding` layers.


## Dataset

[Movie Recommendation System — Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=movies.csv)


## Project Structure

```
CINEMIND/
├── data/
│   ├── raw/                    # Place ratings.csv and movies.csv here
│   └── processed/
│       ├── movies_encoded.csv  # Feature-engineered dataset
│       ├── movies_1.json       # Movie titles with IDs (used by frontend)
│       └── ratings.parquet     # Preprocessed ratings used by the API
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── server.py
│   └── styles.css
├── models/
│   ├── model.py                # Model architecture
│   └── model_weights.pth       # Trained weights
├── notebooks/
│   ├── collab/
│   └── eda/
├── api.py
├── requirements.txt
└── README.md
```


## Setup

Install all dependencies before running anything:

```bash
pip install -r requirements.txt
```




## Data Preprocessing

The raw `movies.csv` required some feature engineering before training:

- **One-hot encoding** applied to the `genres` column
- **Year normalization** to bring movie release years into a consistent scale

The processed data is saved to:

```
data/processed/movies_encoded.csv
```

Movie titles alongside their respective `movieId`s are also extracted for use by the frontend:

```
data/processed/movies_1.json
```




## Running the Project Locally

**1. Add the ratings and movies files**

- Place `ratings.csv` and `movies.csv` inside `data/raw/`.
- Generate the Parquet file used by the API:

```bash
python data/raw/para.py
```

This will create `data/processed/ratings.parquet`.

**2. Start the API server**

From the project root:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 5000
```

**3. Start the frontend server**

In a separate terminal, navigate to the `frontend/` directory and run:

```bash
python server.py
```

**4. Open in your browser**

```
http://localhost:8080/index.html
```


## Model

The recommender uses **Matrix Factorization** users and movies are each represented as learned embedding vectors, and the model predicts ratings from their dot product interaction.

### Training Hardware

Trained on an **NVIDIA RTX 3060 12GB**.

### Hyperparameters

| Parameter | Value |
|---|---|
| Batch Size | 2048 |
| Epochs | 10 |
| Learning Rate | 0.01 |
| Weight Decay | 1e-5 |
| Test Split | 0.2 |

The notebooks in the `notebooks/` directory contain all exploratory data analysis and experimentation done prior to finalizing the model.
