import pandas as pd
import pickle

# retrieve all the sheets names
xls_data = pd.ExcelFile(r"Sample of NIHdata_Set 1.xlsx");
# print(f"sheets names:xls_data:{xls_data.sheet_names}")


# function to remove empty data frame ( no columns and no rows)
def isEmpty(df):
    return len(df.columns.values) == 0 and len(df.index.values) == 0


# A List containing all data frames read from all sheets
sheets_samples = dict()
sheets = xls_data.sheet_names
for sheet_name in sheets:
    print(sheet_name)
    print(r"##############################################################################")
    a_sheet = pd.read_excel(xls_data, sheet_name)
    if not isEmpty(a_sheet):
        sheets_samples[sheet_name] = a_sheet


# retrieve data from data dictionary
# a dict containing data from data dictionary
sheets_dicts = dict()
xls_data_dict = pd.ExcelFile("Data dictionary NIH data_linked dataA.xlsx")
sheets = xls_data_dict.sheet_names
for sheet_name in sheets:
    print(sheet_name)
    a_sheet = pd.read_excel(xls_data_dict, sheet_name)
    print("******************************************************************************")
    if not isEmpty(a_sheet):
        sheets_dicts[sheet_name] = a_sheet


# return the difference of two dict
def compare(dict0, dict1):
    print(f"length of dict0 is{len(dict0)}")
    print(f"length of dict1 is{len(dict1)}")
    count = 0
    for _, element in enumerate(dict0):
        if element in dict1:
            count += 1
    print(f"number of sheets_name in common: {count}")


compare(sheets_samples, sheets_dicts)
print(sheets_dicts)
with open("sample_data.dict", "wb") as sample_data_dict:
    pickle.dump(sheets_samples, sample_data_dict)

with open("dictionary_data.dict", "wb") as dictionary_dict:
    pickle.dump(sheets_dicts, dictionary_dict)








