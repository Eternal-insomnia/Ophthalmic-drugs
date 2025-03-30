from final_filter import filter_and_sort_molecules
import pandas as pd

df = pd.read_csv("sampling.csv")

df_list = df.SMILES.to_list()

result = filter_and_sort_molecules(df_list[:5000])
final_df = pd.DataFrame(result, columns=["SMILES"])

final_df.to_csv("final_smiles.csv")