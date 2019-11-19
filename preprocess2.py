import pandas as pd
import pickle
from typing import Dict
from typing import List
import os

Months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

def is_Date(string: str):
    try:
        if len(str) != 5:
            return False
        st0 = string[:3]
        st1 = string[3:]

        if st0 in Months and st1.isnumeric():
            return True
        return False
    except:
        return False

class MonthYear:
    def __init__(self, monthyear: str):
        month = monthyear[:3]
        year = monthyear[3:]
        if month in Months:
            self.mon = Months.index(month)
        else: 
            raise Exception("month string invalid")
        year = int(year)
        if year < 0 or year > 99:
            raise Exception("year string invalid")
        if year > 50:
            self.year = 1900 + year 
        else:
            self.year = 2000 + year

def minus_MonthYear(moyr0: MonthYear, moyr1: MonthYear):
    return (moyr0.year - moyr1.year) * 12 + moyr0.mon - moyr1.mon

def is_equal_MonthYear(moyr0: MonthYear, moyr1: MonthYear):
    return moyr0.mon == moyr1.mon and moyr0.year == moyr1.year

# read all data from csvs
df_tests = pd.read_csv("li_dhec_tests.csv")
# print("li_dhec_tests.csv")
other_dfs = []
df_rw = pd.read_csv("li_dhec_rw_services.csv")
other_dfs.append(df_rw)
# #print("li_dhec_rw_services.csv")
df_hssc = pd.read_csv("li_dhec_hssc_cohort.csv")
other_dfs.append(df_hssc)
#print("li_dhec_hssc_cohort.csv")
df_cases = pd.read_csv("li_dhec_cases.csv")
other_dfs.append(df_cases)
#print("li_dhec_cases.csv")
df_capss = pd.read_csv("li_dss_capss.csv")
other_dfs.append(df_capss)
# #print("li_dss_capss.csv")
df_chip_client = pd.read_csv("li_dss_chip_client.csv")
other_dfs.append(df_chip_client)
# #print("li_dss_chip_client.csv")
df_chip_part = pd.read_csv("li_dss_chip_participation.csv")
other_dfs.append(df_chip_part)
# #print("li_dss_chip_participation.csv")
df_ub_allpayer = pd.read_csv("li_ub_allpayer.csv")
other_dfs.append(df_ub_allpayer)
# #print("li_ub_allpayer.csv")

# convert tests
def identify_user_rows(df: pd.DataFrame) -> List[int]:
    ans = list()
    if df["RFA_ID"].count() == 0:
        raise Exception("df is empty")
    count = 0
    val = df.loc[count, "RFA_ID"]
    for i, item in df["RFA_ID"].iteritems():
        if item != val:
            val = item
            ans.append(i)
        count += 1
    ans.append(count)
    return ans

groups = identify_user_rows(df_tests)

def convert_df(df: pd.DataFrame) -> List[dict]:
    res = []
    #identify the test dates
    for i, items in df.TEST.iteritems():
        if items.startswith("CD") or items.startswith("VL"):
            if len(res) == 0:
                tmp = dict()
                for name in df.columns.values:
                    tmp[name] = df.loc[i, name]
                res.append(tmp)
            else:
                has_same_time_stamp = False
                for tup in res:
                    date0 = MonthYear(tup["TESTDATE"])
                    date1 = MonthYear(df["TESTDATE"].loc[i])
                    if is_equal_MonthYear(date0, date1):
                        has_same_time_stamp = True
                        break
                if not has_same_time_stamp:
                    tmp = dict()
                    for name in df.columns.values:
                        tmp[name] = df.loc[i, name]
                    res.append(tmp)
    return res

def add_column_time_to_last(li: List[dict]):
    def k(d: dict) -> int:
        s = d["TESTDATE"]
        t = MonthYear(s)
        return t.year*12 + t.mon
    li.sort(key=k)

    for i, dl in enumerate(li):
        da = MonthYear(dl["TESTDATE"])
        if i - 1 < 0:
            ba = da
        else:
            ba = MonthYear(li[i - 1]["TESTDATE"])
        interval = minus_MonthYear(da, ba)
        dl["interval"] = interval


cols = list(df_tests.columns.values)
cols.append("interval")
# check if historical results exit:
df_new = None
for file_entry in os.scandir("."):
    if file_entry.is_file() and file_entry.name == "df_new.pkl":
        f = open(file_entry.path, "rb")
        df_new = pickle.load(f)

if df_new is None:
    df_new = pd.DataFrame(columns=cols)
    start_index = 0
    for i in groups:
        print(f"processing {i}th records...")
        df_ids = df_tests.loc[start_index:i-1, :]
        dicts = convert_df(df_ids)
        add_column_time_to_last(dicts)
        for d in dicts:
            df_new = df_new.append(d, ignore_index=True)
        start_index = i
# save intermediate result
    f = open("df_new.pkl","wb")
    pickle.dump(df_new, f)
    f.close()
df_new.to_csv("df_new.csv")
# stage one finished formatting the tests dataset
########################################################################################################
# the keys are the RFA-id, this is to convert all other date frame to dictionaries.
def create_dict_from_df(df: pd.DataFrame) -> Dict[int, dict]:
    ans = dict()
    for i, item in df.RFA_ID.iteritems():
        if item not in ans:
            row = dict()
            for name in df.columns.values:
                val = df.loc[i, name]
                #Date columns are not added.
                if is_Date(val):
                    continue
                row[name] = df.loc[i, name]
            ans[item] = row
    return ans

# this is to convert tests data frame to a dictionary
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


# this is to merge all the other dicts to the df_new_dict
# retrieve historical result
df_new_dict = None
for file_entry in os.scandir("."):
    if file_entry.is_file() and file_entry.name == "df_new_dict.pkl":
        f = open("df_new_dict.pkl", "wb")
        df_new_dict = pickle.load(f)
        f.close()
if df_new_dict is None:
    print("converting df new...")
    df_new_dict = convert_df_to_dict(df_new)
    col_names = set()
    for df in other_dfs:
        # this is to store all the possible column names
        print("converting other dfs...")
        df_dict = create_dict_from_df(df)
        for RFA_ID, col in df_dict.items():
            print(f"processing RFA_ID:{RFA_ID}...")
            if RFA_ID in df_new_dict:
                for col_name, val in col.items():
                    col_names.add(col_name)
                    for di in df_new_dict[RFA_ID]:
                        for col_n in di:
                            col_names.add(col_n)
                        di[col_name] = val

    # save intermediate results
    f = open("df_new_dict.pkl", "wb")
    pickle.dump(df_new_dict, f)
    f.close()
# creat the final dataframe
print("merging...")
df_final = pd.DataFrame(columns=list(col_names))
for _, dicts in df_new_dict.items():
    for di in dicts:
        df_final = df_final.append(di, ignore_index=True)

print("finishing...")
df_final.to_csv("df_final.csv")
