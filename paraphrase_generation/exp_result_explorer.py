import pandas as pd

filename = r'run_5epoch_results.csv'

df = pd.read_csv(filename)

print(df.sort_values(by="dev_reference_bleu", ascending=False))