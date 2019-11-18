import pandas as pd
from typing import List
Months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

def is_Date(string: str):
    if len(str) != 5:
        return False
    st0 = string[:3]
    st1 = string[3:]

    if st0 in Months and st1.isnumeric():
        return True
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
df_new = pd.DataFrame(columns=cols)
start_index = 0
for i in groups:
    print(f"processing {i}th records...")
    df_ids = df_tests.loc[start_index:i-1, :]
    print(df_ids)
    dicts = convert_df(df_ids)
    add_column_time_to_last(dicts)
    for d in dicts:
        df_new = df_new.append(d, ignore_index=True)
    start_index = i

# binary search a RFA_ID
def binary_search(df: pd.DataFrame, groups: List[int], l: int, r: int, id: int) -> int:
    if r >= l:
        m = l + (r - l) // 2
        idx = groups[m]
        RFA_ID = df.loc[idx, "RFA_ID"]
        if RFA_ID == id:
            return m
        if RFA_ID > id:
            return binary_search(df, groups, l, m - 1, id)
        return binary_search(df, groups, m + 1, r, id)
    return -1

final = []
dims = set()
def concat(li: List[dict], di: dict):
    for key in di:
        val = di[key]
        for redi in li:
            if is_Date(val):
                redi[val] = 0
            else:
                redi[key] = val
        if is_Date(val):
            for redi in li:
                date0 = MonthYear(redi["TESTDATE"])
                date1 = MonthYear(val)

                if 0 <= minus_MonthYear(date0, date1) < redi["interval"]:
                    redi[key] = 1


groups = identify_user_rows(df_new)
groups.insert(0, 0)
for df in other_dfs:
    id_res = dict()
    print("concatenating tables...")
    columns = df.columns.values
    for i, item in df["RFA_ID"].iteritems():
        tp = dict()
        for name in df.columns.values:
            dims.add(name)
            tp[name] = df.loc[i, name]
        if item in id_res:
            idx = id_res
        else:
            idx = binary_search(df_new, groups, 0, len(groups) - 1, item)
            id_res[item] = idx

        if idx == -1:
            continue

        if idx != len(groups) - 1:
            concatee = list()
            for k in range(groups[idx], groups[idx + 1]):
                tmp = dict()
                for name in df_new.columns.values:
                    dims.add(name)
                    tmp[name] = df_new.loc[k, name]
                concatee.append(tmp)
            concat(concatee, tp)

