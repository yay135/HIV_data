# Suppose we have a file indicating all the HIV patients with columns: RFA_ID. File name is "patient.csv".
# And we have a file indicating the testing records of HIV patients with columns: RFA_ID, TESTDATE. File_name is "test.csv".
# execute in command line: "python label_code.py patient.csv test.csv". The program will output all the labels to a new file "labels.csv".

# Type 3-- never been in care
# Type 2-- now out of care
# Type 1-- sometimes in care but sometimes out of care
# Type 0-- always in_care


import pandas as pd
from datetime import datetime as dt
import datetime as datetime
import numpy as np
import sys


def label(test_series, end_date ):

    if len(test_series) == 0:
       return 3
    if end_date -max(test_series)>datetime.timedelta(days=180):
       return 2
    for i in range(len(test_series)-1):
        if test_series[i+1]-test_series[i]> datetime.timedelta(days=180):
           return 1
    return 0

def main():

  id_file = sys.argv[1]
  test_file = sys.argv[2]
  index = pd.read_csv(id_file).RFA_ID.drop_duplicates().tolist()
  r=pd.read_csv(test_file)

  # not very sure about the end of data collection time
  end_date = '01/01/2019'
  end_date = dt.strptime(end_date,'%m/%d/%Y')

  visit_dic = {}
  for i in index:
      s = r[r.RFA_ID==i]
      s = s.TESTDATE.tolist()
      s = [dt.strptime(x,'%m/%d/%Y') for x in s]
      s.sort()
      visit_dic[i] = label(s, end_date)

  labels = pd.DataFrame.from_dict(visit_dic,orient='index',columns=['label'])

  labels.to_csv("labels.csv")

if __name__ == '__main__':

    main()