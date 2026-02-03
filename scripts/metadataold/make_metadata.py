import os, argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Known / normalised labels across both datasets
NORMALISE = {
    'bags': 'dark_circles',
    'bags_under_eyes': 'dark_circles',
    'skin_redness': 'redness',
}
KNOWN = {
    'acne','redness','dark_circles','pigmentation','wrinkles',
    'dryness','oily_skin','blackheads','bags','bags_under_eyes','skin_redness'
}
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def norm_label(lbl: str) -> str:
    l = lbl.strip().lower().replace(' ', '_')
    return NORMALISE.get(l, l)

def find_label_in_parts(parts):
    """Return first known class name found anywhere in the relative path parts."""
    for p in parts:
        p2 = p.strip().lower().replace(' ', '_')
        if p2 in KNOWN:
            return norm_label(p2)
    return None

def scan_recursive(root: str) -> pd.DataFrame:
    rows = []
    root = os.path.abspath(root)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)             # e.g. '123/left/redness/front.jpg'
                parts = rel.split(os.sep)
                label = find_label_in_parts(parts)
                if label is None:
                    # Skip files where we can't infer a label
                    continue
                rows.append((rel, label))
    df = pd.DataFrame(rows, columns=['image','labels'])
    return df

def stratified_split(df: pd.DataFrame, val: float, test: float, seed: int) -> pd.DataFrame:
    """Stratified split by the single label column so each split has all classes."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # First split off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    y = df['labels']
    idx_trainval, idx_test = next(sss1.split(df, y))
    trainval = df.iloc[idx_trainval].reset_index(drop=True)
    test_df  = df.iloc[idx_test].reset_index(drop=True)

    # Then split train/val from trainval
    val_ratio = val / (1.0 - test) if (1.0 - test) > 0 else 0.0
    if val_ratio > 0:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        y_tv = trainval['labels']
        idx_train, idx_val = next(sss2.split(trainval, y_tv))
        train_df = trainval.iloc[idx_train].reset_index(drop=True)
        val_df   = trainval.iloc[idx_val].reset_index(drop=True)
    else:
        train_df = trainval
        val_df   = pd.DataFrame(columns=trainval.columns)

    train_df = train_df.assign(split='train')
    val_df   = val_df.assign(split='val')
    test_df  = test_df.assign(split='test')

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dataset root (any folder depth ok)')
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--val', type=float, default=0.2)
    ap.add_argument('--test', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = scan_recursive(args.root)
    if len(df) == 0:
        raise SystemExit(f'No images found under {args.root}')
    df = stratified_split(df, args.val, args.test, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Wrote {len(df)} rows to {args.out}. Split counts:', df['split'].value_counts().to_dict())
    print('Classes found:', sorted(df['labels'].unique()))
