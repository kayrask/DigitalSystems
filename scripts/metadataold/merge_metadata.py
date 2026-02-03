import argparse, pandas as pd

# Map similar labels to a unified taxonomy for both datasets
NORMALISE = {
    'acne': 'acne',
    'redness': 'redness',
    'skin_redness': 'redness',
    'bags_under_eyes': 'dark_circles',
    'dark_circles': 'dark_circles',
    'pigmentation': 'pigmentation',
    'wrinkles': 'wrinkles',
    'dryness': 'dryness',
    'oily_skin': 'oily_skin',
    'blackheads': 'blackheads'
}

def norm_label(lbl: str) -> str:
    l = lbl.strip().lower().replace(' ', '_')
    return NORMALISE.get(l, l)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_a', required=True)   # from dataset A
    ap.add_argument('--csv_b', required=True)   # from dataset B
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    for df in (df_a, df_b):
        df['labels'] = df['labels'].apply(lambda s: ','.join(sorted({norm_label(x) for x in str(s).split(',')})))
    df = pd.concat([df_a, df_b], ignore_index=True)
    # keep original split columns if present; otherwise, shuffle & rebuild
    if 'split' not in df.columns:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df); n_test = int(n*0.1); n_val = int(n*0.2)
        df['split'] = 'train'
        df.loc[:n_test-1, 'split'] = 'test'
        df.loc[n_test:n_test+n_val-1, 'split'] = 'val'
    df.to_csv(args.out, index=False)
    print(f'Merged -> {args.out} with {len(df)} rows, splits:', df["split"].value_counts().to_dict())
