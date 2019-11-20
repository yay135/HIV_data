import pandas as pd
import os
from typing import List
from typing import Dict
from typing import Tuple
import torch


class Encoder:
    def convert_df_to_dict(self, df: pd.DataFrame) -> Dict[int, List[dict]]:
        ans = dict()
        for i, item in df.RFA_ID.iteritems():
            if item not in ans:
                ans[item] = []
            row = dict()
            for name in df.columns.values:
                row[name] = df.loc[i, name]
            ans[item].append(row)
        return ans
    def __init__(self):
        # open file
        print("reading files...")
        df = None
        for file_entry in os.scandir("."):
            if file_entry.is_file() and file_entry.name == "rfa_interval.m.csv":
                df = pd.read_csv("rfa_interval.m.csv")
        if df is None:
            raise Exception("file rfa_interval not found")
        self.di = self.convert_df_to_dict(df)
        print("finished reading files...")
    # return none if row number is smaller than 6
    def encode(self, id: str, length: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        rfa_id = int(id)
        rows = self.di[rfa_id]
        rows_len = len(rows)
        if rows_len < 6:
             return None
       
        ans = [0 for _ in range(0, length)]
        for i in range(rows_len -1 -1, -1, -1):
            j = i - (rows_len - 1 - length)
            if j < 0:
                break

            t = rows[i]["interval"]
            if t != 1:
                ans[j] = t

        labels = [ans[i] for i in range(1, len(ans))]
        labels.append(rows[-1]["interval"])
        
        matrix = torch.FloatTensor([[ans[i]] for i in range(0, len(ans))])
        labs = torch.LongTensor(labels)
        return matrix, labs

encoder = Encoder()
print(encoder.encode("156141", 10))