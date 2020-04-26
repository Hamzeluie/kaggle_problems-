#
import pandas as pd
from settings import *


data = pd.read_csv(DATA_FE, dtype = {'fullVisitorId': 'str'})
test = pd.read_csv(TEST_FE, dtype = {'fullVisitorId': 'str'})
data_fe = pd.read_csv(DATA_FE_OLIVIER)
test_fe = pd.read_csv(TEST_FE_OLIVIER)

data_combine = pd.merge(data, data_fe, how='outer')
test_combine = pd.merge(test, test_fe, how='outer')
data_combine.drop(columns=['count_pageviews_per_network_domain'], inplace=True)
test_combine.drop(columns=['count_pageviews_per_network_domain'], inplace=True)
data_combine.to_csv(DATA_COMBINE, index=False)
test_combine.to_csv(TEST_COMBINE, index=False)
