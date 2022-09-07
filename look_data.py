import pandas as pd
import numpy as np

df = pd.read_pickle("data/China_A_shares.pandas.dataframe")
print(df)

tech_id_list = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

tech_ary = list()
close_ary = list()
df_len = len(df.index.unique())  # df_len = max_step
for day in range(df_len):
    item = df.loc[day]

    tech_items = [item[tech].values.tolist() for tech in tech_id_list]
    # 把特征展开
    tech_items_flatten = sum(tech_items, [])
    tech_ary.append(tech_items_flatten)

    close_ary.append(item.close)

close_ary = np.array(close_ary)
tech_ary = np.array(tech_ary)
