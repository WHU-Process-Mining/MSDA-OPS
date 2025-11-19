import pandas as pd
from tqdm import tqdm
from river import tree
from collections import deque
from river import stats
import math
from river.drift import ADWIN

from modules.simulator import CASE_ID_KEY, RESOURCE_KEY, START_TIME_KEY, END_TIME_KEY
from utils.time_utils import count_false_hours

def build_training_df_wt(log: pd.DataFrame, res_calendars: dict) -> dict:

    resources = list(res_calendars.keys())

    dict_df_per_res = {res: {'hour': [],
                            'weekday': [],
                            'n. running events': [],
                            'waiting_time': [],
                                                }
                       for res in resources}

    for _, events in log.groupby(CASE_ID_KEY):
        events = events.sort_values([START_TIME_KEY, END_TIME_KEY], kind="mergesort").reset_index(drop=True)
        for i in range(1,len(events)):
            # row_prev = events.iloc[i-1]
            # row_cur  = events.iloc[i]
            res = events.iloc[i][RESOURCE_KEY]
            prev_ts = events.iloc[i-1][END_TIME_KEY]
            start_ts = events.iloc[i][START_TIME_KEY]
            dict_df_per_res[res]['hour'].append(prev_ts.hour)
            dict_df_per_res[res]['weekday'].append(prev_ts.weekday())
            # runing events on the resource
            log_filtered = log[(log[END_TIME_KEY] >= prev_ts) & (log[START_TIME_KEY] < prev_ts)]
            n_active_ev = (log_filtered[RESOURCE_KEY]==res).sum()
            dict_df_per_res[res]['n. running events'].append(n_active_ev)
            dict_df_per_res[res]['waiting_time'].append(max((start_ts - prev_ts).total_seconds()/60 - count_false_hours(res_calendars[res], prev_ts, start_ts)*60 , 0))

    df_per_res = {res: pd.DataFrame(dict_df_per_res[res]) for res in resources}

    return df_per_res

class WaitingTimeModule:

    def __init__(self, initial_log: pd.DataFrame, resource_calendars, grace_period: int = 1000):
        print("Waiting Time Initialization")
        self.grace_period = grace_period
        self.resource_calendars = resource_calendars
        self.waiting_time_distrib, self.min_wt, self.max_wt = self.discover_waiting_time_time_distrib(initial_log, grace_period)
        self.detectors = {res: ADWIN(min_window_length=10, delta=0.05) for res in self.resource_calendars.keys()}

        self.buffers   = {res: deque(maxlen=1000) for res in self.resource_calendars.keys()}
        self.active_intervals = {res: deque(maxlen=1000) for res in self.resource_calendars.keys()}
        self.err_window = {res: deque(maxlen=10) for res in self.resource_calendars.keys()}

        self.rebuilding_time = 0

    def _concurrency_at(self, res, ts):
        return sum(1 for (s, e) in self.active_intervals.get(res, ()) if s <= ts < e)

    def discover_waiting_time_time_distrib(self, log: pd.DataFrame, grace_period: int = 1000, max_depth: int = 5):
        resources = list(self.resource_calendars.keys())
        df_per_res = build_training_df_wt(log, self.resource_calendars)

        models_res = dict()

        min_wt = dict()
        max_wt = dict()

        for res in tqdm(resources):
            df_res = df_per_res[res]

            models_res[res] = tree.HoeffdingAdaptiveTreeRegressor(seed=72, leaf_prediction="mean", max_depth=max_depth, grace_period=grace_period)
            
            min_wt[res] = df_res["waiting_time"].min()
            max_wt[res] = df_res["waiting_time"].max()

            for _, row in df_res.iterrows():
                X_row = row.drop('waiting_time').to_dict()
                y_row = row['waiting_time']
                models_res[res].learn_one(X_row, y_row)

        return models_res, min_wt, max_wt
    
    def get_waiting_time(self, resource, cur_ts, running_event_num):
        waiting_time = self.waiting_time_distrib[resource].predict_one({'hour': cur_ts.hour, 
                                                                            'weekday': cur_ts.weekday(), 
                                                                            'n. running events': running_event_num})
        return waiting_time
    
    def update(self, trace: pd.DataFrame, resource_calendars: dict):
        self.resource_calendars = resource_calendars
        row = trace.iloc[-1]
        res = row[RESOURCE_KEY]
        start_ts = row[START_TIME_KEY]
        end_ts   = row[END_TIME_KEY]

        if res not in self.waiting_time_distrib:
            self.waiting_time_distrib[res] = tree.HoeffdingAdaptiveTreeRegressor(
                leaf_prediction="mean", max_depth=5, grace_period=self.grace_period, seed=72
            )
        if res not in self.detectors:
            self.detectors[res] = ADWIN(min_window_length=10, delta=0.05)
        if res not in self.buffers:
            self.buffers[res] = deque(maxlen=1000)
        if res not in self.active_intervals:
            self.active_intervals[res] = deque(maxlen=1000)
        if res not in self.min_wt:
            self.min_wt[res] = math.inf
        if res not in self.max_wt:
            self.max_wt[res] = -math.inf
        if res not in self.err_window:
            self.err_window[res] = deque(maxlen=10)

        if len(trace)>1: # at least two event
            events = trace.sort_values([START_TIME_KEY, END_TIME_KEY], kind="mergesort").reset_index(drop=True)

            prev = events.iloc[-2]
            cur  = events.iloc[-1]

            res = cur[RESOURCE_KEY]         
            prev_ts = prev[END_TIME_KEY]    
            start_ts = cur[START_TIME_KEY]

            n_active_ev = self._concurrency_at(res, prev_ts)

            X = {
                'hour': prev_ts.hour,
                'weekday': prev_ts.weekday(),
                'n. running events': n_active_ev
            }
                
                
            y_hat = self.waiting_time_distrib[res].predict_one(X)

            cal_res = self.resource_calendars.get(res, {wd: {h: True for h in range(24)} for wd in range(7)})
            off_duty_h = count_false_hours(cal_res, prev_ts, start_ts)  # 返回小时数
            y = max((start_ts - prev_ts).total_seconds()/60.0 - off_duty_h*60.0, 0.0)

            # 维护 min/max
            self.min_wt[res] = min(self.min_wt[res], y)
            self.max_wt[res] = max(self.max_wt[res], y)

            # 近窗样本缓冲
            self.buffers[res].append((X.copy(), y))
            self.active_intervals[res].append((start_ts, end_ts))

            err   = abs(y - y_hat)
            
            self.err_window[res].append(err)
            win_mae = sum(self.err_window[res]) / len(self.err_window[res])
            self.detectors[res].update(err)

            error_flag = False
            if len(self.err_window[res]) == self.err_window[res].maxlen and win_mae > 240:
                error_flag = True

            if self.detectors[res].drift_detected or error_flag:
                self.rebuilding_time += 1
                if self.detectors[res].drift_detected:
                    # print(f"Drift")
                    width = min(int(self.detectors[res].width), len(self.err_window[res]))
                    recent = list(self.buffers[res])[-width:]
                    self.detectors[res] = ADWIN(min_window_length=10, delta=0.05)
                else:
                    # print(f"ERROR")
                    recent = [self.buffers[res][-1]]
                self.err_window[res].clear()
                # print(f"Rebuliding Waiting Time Module for res: {res} at {start_ts}")
                new_model = tree.HoeffdingAdaptiveTreeRegressor(leaf_prediction="mean", max_depth=5, seed=72, grace_period=self.grace_period)
                for Xb, yb in recent:
                    new_model.learn_one(Xb, yb)
                self.waiting_time_distrib[res] = new_model
            else:
                # 正常增量学习
                self.waiting_time_distrib[res].learn_one(X, y)

                
