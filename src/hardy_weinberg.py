import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

def calculate_hwe(df, col):
    counts = df[col].value_counts()
    norm = counts.get('normal', 0)
    het = counts.get('heterozygot', 0)
    mut = counts.get('mutant', 0)
    total = norm + het + mut
    if total == 0:
        return None

    p = (2 * norm + het) / (2 * total)
    q = 1 - p
    expected = np.array([p**2, 2*p*q, q**2]) * total
    observed = np.array([norm, het, mut])
    valid = expected > 0

    if valid.sum() <= 1:
        chi2, pval, dfree, hwe = np.nan, np.nan, 0, "Undetermined"
    else:
        chi2, pval = stats.chisquare(observed[valid], expected[valid])
        dfree = max(1, valid.sum() - 2)
        hwe = "Yes" if pval > 0.05 else "No"

    return {
        "Mutation": col,
        "Total": total,
        "Observed": dict(zip(['Normal', 'Het', 'Mut'], observed)),
        "Expected": dict(zip(['Normal', 'Het', 'Mut'], expected)),
        "Allele frequencies": (p, q),
        "Chi2": chi2,
        "p-value": pval,
        "DF": dfree,
        "In HWE": hwe
    }

def plot_distribution(result, outdir):
    labels = ['Normal', 'Het', 'Mut']
    obs = list(result['Observed'].values())
    exp = list(result['Expected'].values())
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, obs, width=0.4, label='Observed')
    plt.bar(x + 0.2, exp, width=0.4, label='Expected')
    plt.xticks(x, labels)
    plt.title(f"HWE: {result['Mutation']} (p={result['p-value']:.4f})")
    plt.legend()

    if outdir:
        Path(outdir).mkdir(exist_ok=True)
        fname = Path(outdir) / f"hwe_{result['Mutation'].replace(' ', '_')}.png"
        plt.savefig(fname)
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="../res/SSBU25_dataset_samuel.xlsx")
    parser.add_argument("--plots", "-g", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_excel(args.path)
    hfe_cols = [col for col in df.columns if 'hfe' in col.lower()]

    print(f"Found columns: {hfe_cols}\n")

    results = []
    for col in hfe_cols:
        print(f"--- {col} ---")
        res = calculate_hwe(df, col)
        if not res:
            print("No data.")
            continue
        results.append(res)
        print(f"Observed: {res['Observed']}")
        print(f"Expected: {res['Expected']}")
        print(f"p = {res['p-value']:.4f}, In HWE: {res['In HWE']}\n")
        if args.plots:
            plot_distribution(res, args.output)

if __name__ == "__main__":
    sys.stdout = open("../res/hardy-samuel.txt", "w", encoding="utf-8")
    main()