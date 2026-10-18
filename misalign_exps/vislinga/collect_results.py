"""Merge the per-model JSONs written by eval_vislinga.py into one table (printed and saved as CSV)."""
import argparse
import json
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
args = p.parse_args()

rows = []
for f in sorted(args.results.glob('*.json')):
    r = json.loads(f.read_text())
    rows.append({'modality': r['modality'], 'model': r['model'], 'params_M': round(r['n_params'] / 1e6),
                 **r['scores']})

df = pd.DataFrame(rows).sort_values(['modality', 'top10'], ascending=[False, False])
df.to_csv(args.results / 'summary.csv', index=False)
print(df.to_string(index=False, float_format='%.3f'))
print(f'\n{len(df)} models')
