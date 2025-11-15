import pandas as pd
from river.drift import ADWIN
from river import tree

from modules.simulator import CASE_ID_KEY, START_TIME_KEY, END_TIME_KEY
from utils.time_utils import count_false_hours, add_minutes_with_calendar, cal_error_with_calendar



def discover_arrival_calendar(log: pd.DataFrame, thr_h: float = 0.0, thr_wd: float = 0.0) -> dict:

    N_events_per_hour = {wd: {h: 0 for h in range(24)} for wd in range(7)}
    
    for _, events in log.groupby(CASE_ID_KEY):
        first_idx = events[START_TIME_KEY].idxmin()
        ts = events.loc[first_idx, START_TIME_KEY]
        N_events_per_hour[ts.weekday()][ts.hour] += 1
        ts = events.loc[first_idx, END_TIME_KEY]
        N_events_per_hour[ts.weekday()][ts.hour] += 1

    N_events_per_hour_perc = {wd: {h: 0 for h in range(24)} for wd in range(7)}

    for weekday in range(7):
        for h in range(24):
            if sum(N_events_per_hour[weekday].values()):
                N_events_per_hour_perc[weekday][h] = N_events_per_hour[weekday][h] / sum(N_events_per_hour[weekday].values())
            else:
                N_events_per_hour_perc[weekday][h] = 0

    N_events_per_wd = {wd: 0 for wd in range(7)}

    for weekday in range(7):
        N_events_per_wd[weekday] += sum(N_events_per_hour[weekday].values())

    N_events_per_wd_perc = {wd: N_events_per_wd[wd]/sum(N_events_per_wd.values()) if sum(N_events_per_wd.values()) else 0 for wd in range(7)}

    calendar_wd_hour = {wd: {h: False for h in range(24)} for wd in range(7)}

    for wd in range(7):
        if N_events_per_wd_perc[wd] <= thr_wd:
            continue
        else:
            for hour in range(24):
                calendar_wd_hour[wd][hour] = N_events_per_hour_perc[wd][hour] > thr_h

    return calendar_wd_hour

def build_training_df_arrival(
        log: pd.DataFrame, 
        calendar_arrival: dict
    ) -> pd.DataFrame:

    dict_df = {'hour': []} | {'weekday': []} | {'arrival_time': []}

    arrivals = (log.groupby(CASE_ID_KEY, sort=False)[START_TIME_KEY]
              .min()
              .sort_values()
              .reset_index(name='case_start_time'))

    for i in range(len(arrivals) - 1):
        cur_ts  = arrivals.loc[i,   'case_start_time']
        next_ts = arrivals.loc[i+1, 'case_start_time']

        dict_df['hour'].append(cur_ts.hour)
        dict_df['weekday'].append(cur_ts.weekday())
        # minus non-working hours
        off = max(count_false_hours(calendar_arrival, cur_ts, next_ts), 0.0)
        dt  = (next_ts - cur_ts).total_seconds()/60.0
        ar_time = max(dt - off*60.0, 0.0) #min
        dict_df['arrival_time'].append(ar_time)   
    
    df = pd.DataFrame(dict_df)
    
    return df

class ArrivalTimeModule:

    def __init__(self, initial_log: pd.DataFrame, grace_period: int = 1000):
        print("Arrival Time Model Initialization...")
        self.grace_period = grace_period
        self.arrival_calendar = discover_arrival_calendar(initial_log)
        self.arrival_time_model, self.min_at, self.max_at = self.discover_arrival_model(initial_log, grace_period)
        self.last_arrival_time = max(initial_log.groupby(CASE_ID_KEY)[START_TIME_KEY].min())
        self.arrival_times = []
        self.case_id = 0
        last_arrival_event = initial_log[initial_log[START_TIME_KEY]==self.last_arrival_time].iloc[-1]
        self.update_rows = [{CASE_ID_KEY: f'case_{self.case_id}', START_TIME_KEY: self.last_arrival_time, END_TIME_KEY: last_arrival_event[END_TIME_KEY]}]
        self.update_num = 0
        self.detector = ADWIN(min_window_length=100)

    def discover_arrival_model(self, log: pd.DataFrame, grace_period: int = 1000, max_depth: int=5):
        df = build_training_df_arrival(log, self.arrival_calendar)

        arrival_model = tree.HoeffdingAdaptiveTreeRegressor(leaf_prediction="mean", max_depth=max_depth, seed=72, grace_period=grace_period)

        for _, row in df.iterrows():
            X_row = row.drop('arrival_time').to_dict()
            y_row = row['arrival_time']
            arrival_model.learn_one(X_row, y_row)
        
        min_at = df["arrival_time"].min()
        max_at = df["arrival_time"].max()

        return arrival_model, min_at, max_at
    
    def update_arrival_time_model(self, log: pd.DataFrame):
        df = build_training_df_arrival(log, self.arrival_calendar)
        for _, row in df.iterrows():
            X_row = row.drop('arrival_time').to_dict()
            y_row = row['arrival_time']
            self.arrival_time_model.learn_one(X_row, y_row)
        
        self.min_at = min(self.min_at, df["arrival_time"].min())
        self.max_at = max(self.max_at, df["arrival_time"].max())


    def get_arrival_time(self, trace_first_event):
        self.case_id += 1
        real_trace_time = trace_first_event[START_TIME_KEY]
        delta = self.arrival_time_model.predict_one({'hour': self.last_arrival_time.hour,
                                                           'weekday': self.last_arrival_time.weekday()})
        arrival_delta = max(self.min_at, min(delta, self.max_at))
        predict_arrival_time = add_minutes_with_calendar(self.last_arrival_time, arrival_delta, self.arrival_calendar)

        self.arrival_times.append(real_trace_time)
        
        self.update_rows.append({CASE_ID_KEY: f'case_{self.case_id}', START_TIME_KEY: real_trace_time, END_TIME_KEY: trace_first_event[END_TIME_KEY]})

        is_arrival_period = bool(self.arrival_calendar[real_trace_time.weekday()][real_trace_time.hour])
        # 1 mismatch
        mismatch = int(not is_arrival_period)
        if mismatch:
            # print(f"Update Arrival Calendar at {real_trace_time}")
            self.arrival_calendar[real_trace_time.weekday()][real_trace_time.hour]=True

        real_last_arrival = self.arrival_times[-2] if len(self.arrival_times)>1 else self.last_arrival_time
        real_predict_delta = self.arrival_time_model.predict_one({'hour': real_last_arrival.hour,
                                                            'weekday': real_last_arrival.weekday()})
        real_predict_arrival = add_minutes_with_calendar(real_last_arrival, real_predict_delta, self.arrival_calendar)

        err_min = cal_error_with_calendar(real_predict_arrival, real_trace_time, self.arrival_calendar)
        rng = max(self.max_at - self.min_at, 1e-6)
        x = min(err_min / rng, 1.0)
        self.detector.update(err_min)

        if self.detector.drift_detected:# retrain
            # print(f"Update Arrival Time Model at {real_trace_time}")
            self.update_num += 1
            update_log = self.update_rows[-int(self.detector.width):]
            update_log = pd.DataFrame(update_log)
            self.arrival_time_model, self.min_at, self.max_at = self.discover_arrival_model(update_log)
            self.update_rows = [{CASE_ID_KEY: f'case_{self.case_id}', START_TIME_KEY: real_trace_time, END_TIME_KEY: trace_first_event[END_TIME_KEY]}]
        else: # single sample increment learning
            update_log = pd.DataFrame(self.update_rows[-2:])
            self.update_arrival_time_model(update_log)
            
        if len(self.arrival_times)==1: # first arrival trace
            predict_arrival_time = real_trace_time
        self.last_arrival_time = predict_arrival_time
        return predict_arrival_time
