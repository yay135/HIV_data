ls0 = {'POVERTY', 'PAYMENT', 'SERVICE_CATEGORY', 'RFA_ID'}
ls1 = {'EDLVL', 'MARICODE', 'RESIDENT'}
ls2 = {'PPOA', 'PAYOR1', 'DISP', 'ADM_TYPE', 'DRIVE_DISTANCE', 'MDC'}
# contains features that are time-series related.
set0 = {'PAYMENT', 'SERVICE_CATEGORY', 'PPOA', 'PAYOR1', 'DISP', 'ADM_TYPE'}
set1 = {'DRIVE_DISTANCE'}
di = {'li_dhec_cases' : "all", 'li_dhec_hssc_cohort' : 'all', 
'li_dhec_rw_services' : ls0, 'li_dss_chip_client' : ls1, 'li_ub_allpayer' : ls2}

def get():
    return di, set0, set1