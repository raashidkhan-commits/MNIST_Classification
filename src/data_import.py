from pathlib import Path
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

out_dir = Path(__file__).resolve().parents[1] / "data"
out_dir.mkdir(parents=True, exist_ok=True)

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# ensure numeric labels
y = np.array(y, dtype=np.int8).reshape(-1)

pd.DataFrame(X).to_csv(out_dir / "mnist_data.csv", index=False, header=False)
pd.Series(y).to_csv(out_dir / "mnist_target.csv", index=False, header=False)

print("Saved:", out_dir / "mnist_data.csv", out_dir / "mnist_target.csv")
