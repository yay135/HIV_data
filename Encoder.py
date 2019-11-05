import pandas as pd
import numpy as np
import pickle

with open("sample_data.dict", "rb") as sample_dict_file:
    sample_dict = pickle.load(sample_dict_file)

with open("dictionary_data.dict", "rb") as dictionary_data_file:
    dictionary_dict = pickle.load(dictionary_data_file)


def str_to_alpha(word):
    my_str =""
    for i in range(0, len(word)):
        if word[i].isalpha():
            my_str += word[i]
    return my_str.lower()


def query_data_frame(df, label):
    print("querying " + str(label) + "...")
    ans = []
    for idx , row in df.iterrows():
        count = 0
        for _ , val in row.iteritems():
            if str_to_alpha(str(val)) == str_to_alpha(str(label)):
                ans.append(df.iloc[idx, count + 1])
                ans.append(df.iloc[idx, count + 2])
            count += 1

    return ans


# refine dictionary dict
refined_dictionary = dict()
for _, sheet_name in enumerate(dictionary_dict):
    label_dict = dict()
    df = dictionary_dict[sheet_name]
    if sheet_name not in sample_dict:
        break
    columns_sample = sample_dict[sheet_name].columns.values
    for label in columns_sample:
        ans = query_data_frame(df, label)
        if len(ans) == 0:
            print(str(label) + " in " + sheet_name + " not found in data dictionary!")
        label_dict[label] = ans
    refined_dictionary[sheet_name] = label_dict


print(refined_dictionary)


# inplace encoder
def count_category(series):
    tmp_dict = dict()
    count = 0
    for _, item in series.iteritems():
        item = str(item)
        if item not in tmp_dict:
            tmp_dict[item] = count
            count += 1
    return tmp_dict


# category value encoder
def cat_encoder(df, type_dict):
    for column_name in df.columns.values:
        type = type_dict[column_name][0]
        if type != "char":
            continue
        a_column = df[column_name]
        cat_dict = count_category(a_column)

        for idx, item in df[column_name].iteritems():
            item = str(item)
            df.loc[idx, column_name] = cat_dict[item]











