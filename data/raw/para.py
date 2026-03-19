# save as convert_ratings_to_parquet.py

import pandas as pd
from pathlib import Path

# paths (adjust if your structure is different)
csv_path = Path(r"C:\Users\jaken\Pictures\CineMind\data\raw\ratings.csv")
parquet_path = Path(r"C:\Users\jaken\Pictures\CineMind\data\processed\ratings.parquet")


# read CSV
df = pd.read_csv(csv_path)

# make sure output directory exists
parquet_path.parent.mkdir(parents=True, exist_ok=True)

# write Parquet (using pyarrow engine)
df.to_parquet(parquet_path, index=False, engine="pyarrow")
print(f"Saved Parquet to {parquet_path.resolve()}")

