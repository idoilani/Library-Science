import pandas as pd
import numpy as np
from dateutil.parser import parse
import datetime
from collections import OrderedDict

PROJECT_DIR = r'/Users/Gal/Documents/Repositories/Workshop-in-Data-Science'


# Changes z-time to epoch timestamp:  E.G. 2008-09-16T21:40:29Z -->  '1210185365'
def zdt2epoch(value):
    value = value.replace('Z', '+00:00')
    d = parse(value)
    return d


def add_time_diff_features(merged_df, is_train=True):
    # convert to datetime
    merged_df['CreationDateAnswerEpoch'] = merged_df['CreationDate_ans'].apply(lambda x: zdt2epoch(x))
    merged_df['CreationDateQuestionEpoch'] = merged_df['CreationDate_qus'].apply(lambda x: zdt2epoch(x))

    print "In add time diff"
    merged_df['timeDiff'] = (merged_df.CreationDateAnswerEpoch - merged_df.CreationDateQuestionEpoch).apply(datetime.timedelta.total_seconds)
    # Find 10 percentiles for bucketing:
    quantiles = merged_df['timeDiff'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]).values.tolist()

    q2min_map = dict()
    for parentId, df in merged_df.groupby('ParentId'):
        q2min_map[parentId] = df['timeDiff'].min()

    def key_fallback(key):
        '''
        Make sure a key (question) have a value (min response time).
        If not - write (-1) as the minimal response time
        '''
        try:
            val = q2min_map[key]
            return val
        except KeyError:
            return -1

    def bucketize_percentile(diff):
        '''
        Bucktize the diffs by the percentiles of the data.
        '''
        for i, val in enumerate(quantiles):
            if diff <= val:
                return i

    def bucketize_timepart(diff):
        '''
        Bucktize the diffs by conventional time intervals like hour, day, week, month, etc.. .
        '''
        times = dict(
            quarterhour=15 * 60,
            halfhour=60 * 30,
            hour=60 * 60,
            day=(60 * 60) * 24,
            threedays=3 * ((60 * 60) * 24),
            twoweeks=2 * (7 * ((60 * 60) * 24)),
            month=4 * 7 * ((60 * 60) * 24),
            threemonths=3 * 4 * 7 * ((60 * 60) * 24),
            halfyear=6 * 4 * 7 * ((60 * 60) * 24),
            year=12 * 4 * 7 * ((60 * 60) * 24),
            threeyears=3 * 12 * 4 * 7 * ((60 * 60) * 24))
        ordered_times = OrderedDict(sorted(times.items(), key=lambda t: t[1]))
        for key in ordered_times.keys():
            if diff <= times[key]:
                return key
        return "more_than_3_years"

    merged_df['MinResponseTime'] = merged_df['Id_qus'].apply(lambda pid: key_fallback(pid))
    merged_df['diff_percentile_bucket'] = merged_df['timeDiff'].apply(lambda x: bucketize_percentile(x))
    merged_df['diff_interval_bucket'] = merged_df['timeDiff'].apply(lambda x: bucketize_timepart(x))

    if is_train:
        merged_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/train_with_time_diff.csv')
        return merged_df
    else:
        merged_df.to_csv(r'C:/Users/Gal/Documents/Library-Science/data/test_with_time_diff.csv')
        return merged_df
