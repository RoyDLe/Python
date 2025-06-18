from pathlib import Path

import pandas as pd

filename = "penguins.csv"
app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / filename)
