import pandas as pd
import os
from typing import Dict, List
import label_dictionary as ld

df_f = pd.DataFrame()

df_labels = pd.read_csv("lables.csv")

class Encoder:
    def __init__(self, path:str, features:list, special_features:set, mean_features:set):
        self.features = features
        self.special_features = special_features
        self.mean_features = mean_features
        self.df = pd.read_csv(path)
        self.di = {}
        if self.features is None:
            self.features = self.df.columns.values
        if len(self.special_features) != 0:
            for feature in special_features:
                record = {}
                col = self.df[feature]
                for _, val in col.iteritems():
                    record[val] = 0
                self.di[feature] = record

    def select_id_categorize(self, id:int) -> Dict:
        selected = self.df.loc[self.df["RFA_ID" == id], self.features]
        dii = {} 
        for feature in self.features:
            if feature in self.special_features:
                col = selected[feature]
                record = self.di[feature].copy()
                for _, val in col.iteritems():
                    record[val] = 1
                for key, val in record.iteritems():
                    dii[key] = val
            elif feature in self.mean_features:
                dii[feature] = selected[feature].mean()
            else:
                dii[feature] = selected.loc[selected.index.values[0], feature]
        return dii
            
fe_dict, special_features, mean_features = ld.get()
ls_encoder = []
for file_name, features in fe_dict.items():
    path = file_name + '.csv'
    if features == 'all':
        features = None
    encoder = Encoder(path, features, special_features, mean_features)
    ls_encoder.append(encoder)

for index, row in df_labels.iterrows():
    id = df_labels[index, "RFA_ID"]
    label = df_labels[index, "label"]
    print(f"processing RFA_ID {id} ...")
    f_di = {}
    f_di[id] = label
    for encoder in ls_encoder:
        di = encoder.mean_features(id)
        f_di.update(di)

    df_f = df_f.append(f_di, ignore_index=True)

df_f.to_csv("merged.csv")