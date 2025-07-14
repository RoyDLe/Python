from pathlib import Path

import pandas as pd

filename = "aviation.csv"
app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / filename)
