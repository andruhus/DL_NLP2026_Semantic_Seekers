import pandas as pd

def sample_csv(input_path, output_path, fraction=0.05, seed=11711):
    # CSV laden
    df = pd.read_csv(input_path)

    # 5% der Daten ziehen (random, aber reproduzierbar)
    df_small = df.sample(frac=fraction, random_state=seed)

    # neue CSV speichern
    df_small.to_csv(output_path, index=False)

    print(f"Original size: {len(df)} rows")
    print(f"Sampled size:  {len(df_small)} rows")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    sample_csv("data/allnli-train.csv", "data/allnli-train-small.csv")
    sample_csv("data/allnli-dev.csv","data/allnli-dev-small.csv")
