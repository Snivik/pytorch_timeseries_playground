from datetime import datetime

import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex


class SyntheticDataGenerator:

    dates: DatetimeIndex = None

    def generate_data(self) -> [float]:
        '''
        Generates values for timeseries. Just calls
        a generator for every minute or so
        :return:
        '''

        values_raw: [float] = []

        index = 0
        for ts in self.dates:
            values_raw.append(self._get_values_at_ts(ts, index))
            index += 1

        return values_raw

    def _get_values_at_ts(self, ts: datetime, index: int) -> float:
        '''
        Generates a datapoint for a timestamp
        :param ts: datetime timestamp
        :param index:
        :return:
        '''

        # Some random logic so that pycharm doesn't complain. This method is to be overwritten
        return ts.day + float(index) / len(self.dates) * 100


