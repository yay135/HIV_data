# this is to convert tests data frame to a dictionary
from typing import Dict, List
import pandas as pd

def convert_df_to_dict(df: pd.DataFrame) -> Dict[int, List[dict]]:
    ans = dict()
    for i, item in df.RFA_ID.iteritems():
        if item not in ans:
           ans[item] = []
        row = dict()
        for name in df.columns.values:
            row[name] = df.loc[i, name]
        ans[item].append(row)
    return ans

df = pd.read_csv("rfa_interval.csv")

labels = pd.DataFrame(columns=["RFA_ID", "labels"])
di = convert_df_to_dict(df)
for rfa_id in di:
    ls = di[rfa_id]
    is_in_care = True
    for row in ls:
        if row["interval_r"] > 6:
            labels = labels.append({"RFA_ID":rfa_id, "labels":0}, ignore_index=True)
            is_in_care = False
            break
    if is_in_care:
        labels = labels.append({"RFA_ID":rfa_id, "labels":1}, ignore_index=True)

labels.to_csv("labels.csv")

