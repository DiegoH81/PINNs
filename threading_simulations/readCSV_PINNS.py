import pandas as pd

df_sir = pd.read_csv("C:\\Users\\diego\\OneDrive\\Escritorio\\Diego\\INVESTIGACION\\PINNs\\threading_simulations\\results_sir.csv")

grouped_sir = (
    df_sir
    .groupby(["points", "balanced", "noise"], as_index=False)
    .mean(numeric_only=True)
)

print("SIR")
print(grouped_sir)

#----------------------------<------------------>----------------------------

df_MT = pd.read_csv("C:\\Users\\diego\\OneDrive\\Escritorio\\Diego\\INVESTIGACION\\PINNs\\threading_simulations\\results_mt.csv")

grouped_MT = (
    df_MT
    .groupby(["points", "balanced", "noise"], as_index=False)
    .mean(numeric_only=True)
)

print("Maki-Thompson")
print(grouped_MT)