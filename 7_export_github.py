import pandas as pd
import numpy as np

df1=pd.read_csv('./work/StormEvents.csv', low_memory=False)
df2=pd.read_csv('./work/StormEvents_fe_ep_augmentation_fin.csv',low_memory=False)

chunks1=np.array_split(df1, indices_or_sections=4)
for i, chunk in enumerate(chunks1):
    chunk.to_parquet(f'./Storm_Events_GitHub/data/StormEvents_part_{i+1}.parquet', engine='pyarrow', compression='zstd')
    
chunks2=np.array_split(df2, indices_or_sections=10)
for i, chunk in enumerate(chunks2):
    chunk.to_parquet(f'./Storm_Events_GitHub/data/StormEvents_fe_ep_augmentation_fin_part_{i+1}.parquet', engine='pyarrow', compression='zstd')