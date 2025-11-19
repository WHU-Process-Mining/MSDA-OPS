import pandas as pd
from tqdm import tqdm
from collections import deque
from river import stats
import math
from river import tree
from river.drift import ADWIN
from modules.simulator import CASE_ID_KEY, ACTIVITY_KEY, RESOURCE_KEY, START_TIME_KEY, END_TIME_KEY
from utils.time_utils import count_false_hours

def build_training_df_ex(
        log: pd.DataFrame, 
        activity_labels: list, 
        res_calendars: dict,
    ) -> pd.DataFrame:
    
    dict_df = {'activity_executed': []} | {
               'resource': []} | {
               act: [] for act in activity_labels} |{
               'hour': []} | {'weekday': []} | {
               'execution_time': []}

    for _, events in log.groupby(CASE_ID_KEY):

        trace_history = {a: 0 for a in activity_labels}
        events = events.sort_values([START_TIME_KEY, END_TIME_KEY], kind="mergesort")
        for _, event in events.iterrows():

            if event[ACTIVITY_KEY] not in activity_labels:
                continue
            
            for a in activity_labels:
                dict_df[a].append(trace_history[a])
        
            act_executed = event[ACTIVITY_KEY]
            dict_df['activity_executed'].append(act_executed)

            trace_history[act_executed] += 1

            start_ts = event[START_TIME_KEY]
            end_ts = event[END_TIME_KEY]

            res = event[RESOURCE_KEY]

            dict_df['resource'].append(res)

            dict_df['hour'].append(start_ts.hour)
            dict_df['weekday'].append(start_ts.weekday())

            ex_time = max((end_ts - start_ts).total_seconds()/60 - count_false_hours(res_calendars[res], start_ts, end_ts)*60, 0)
            dict_df['execution_time'].append(ex_time)
        
    df = pd.DataFrame(dict_df)
    # resource one-hot
    resources = list(res_calendars.keys())
    res_dummies = pd.get_dummies(df['resource'])
    res_dummies = res_dummies.reindex(columns=resources, fill_value=0)
    res_dummies.columns = [f"resource = {r}" for r in resources]
    df = pd.concat(
        [df.drop(columns=['resource']), res_dummies],
        axis=1
    )

    return df

class ExcutionTimeModule:

    def __init__(self, initial_log: pd.DataFrame, resource_calendars, grace_period: int = 1000):
        print("Execution Time Module Initialization")
        self.grace_period = grace_period
        self.activities = list(initial_log[ACTIVITY_KEY].unique())
        self.resource_calendars = resource_calendars
        self.excution_time_distrib, self.min_et, self.max_et = self.discover_excution_time_distrib(initial_log, grace_period)
        self.detectors = {act: ADWIN(min_window_length=10, delta=0.05) for act in self.activities}
        self.buffers   = {act: deque(maxlen=1000) for act in self.activities}
        self.err_window = {act: deque(maxlen=10) for act in self.activities}
        self.rebuilding_time = 0
        

    def discover_excution_time_distrib(self, log: pd.DataFrame, grace_period: int = 1000, max_depth: int = 5) -> dict:
        df = build_training_df_ex(log, self.activities, self.resource_calendars)
        
        models_act = dict()

        min_et = dict()
        max_et = dict()

        for act in tqdm(self.activities):

            df_act = df[df['activity_executed'] == act].iloc[:,1:]

            models_act[act] = tree.HoeffdingAdaptiveTreeRegressor(max_depth=max_depth, leaf_prediction="mean", seed=72, grace_period=grace_period)

            for _, row in df_act.iterrows():
                X_row = row.drop('execution_time').to_dict()
                y_row = row['execution_time']
                models_act[act].learn_one(X_row, y_row)

            min_et[act] = df_act["execution_time"].min()
            max_et[act] = df_act["execution_time"].max()
        
        return models_act, min_et, max_et
    
    def get_execution_time(self, activity, resource, timestamp, historical_act: list):
        
        historical_active_act = {l: 0 for l in self.activities}

        for act in historical_act:
            historical_active_act[act] += 1
        
        execution_time = self.excution_time_distrib[activity].predict_one({'resource = '+res: (res == resource)*1 for res in self.resource_calendars.keys()} | 
                                                                                    historical_active_act |{'hour': timestamp.hour, 'weekday': timestamp.weekday()})
        execution_time = max(self.min_et[activity], min(execution_time, self.max_et[activity]))
        return execution_time
    
    def update(self, trace: pd.DataFrame, resource_calendars):
        self.resource_calendars = resource_calendars

        row = trace.iloc[-1]
        act = row[ACTIVITY_KEY]
        res = row[RESOURCE_KEY]
        start_ts = row[START_TIME_KEY]
        end_ts   = row[END_TIME_KEY]

        if act not in self.activities:
            self.activities.append(act)
        if act not in self.excution_time_distrib:
            self.excution_time_distrib[act] = tree.HoeffdingAdaptiveTreeRegressor(
                leaf_prediction="mean", max_depth=5, seed=72, grace_period=self.grace_period, 
            )
        
        self.detectors.setdefault(act, ADWIN(min_window_length=10, delta=0.05))
        self.buffers.setdefault(act, deque(maxlen=1000))
        self.err_window.setdefault(act, deque(maxlen=10))

        self.min_et.setdefault(act, math.inf)
        self.max_et.setdefault(act, -math.inf)

        hist_counts = {a: 0 for a in self.activities}
        for a in trace.iloc[:-1][ACTIVITY_KEY]:
            hist_counts[a] += 1

        res_one_hot = {'resource = ' + r: int(r == res) for r in self.resource_calendars.keys()}
        time_feats = {'hour': start_ts.hour, 'weekday': start_ts.weekday()}

        X = {}
        X.update(hist_counts)
        X.update(res_one_hot)
        X.update(time_feats)

        # min
        y_hat = float(self.excution_time_distrib[act].predict_one(X))

        cal = self.resource_calendars.get(res, {wd: {h: True for h in range(24)} for wd in range(7)})
        false_hours = count_false_hours(cal, start_ts, end_ts)
        y = max((end_ts - start_ts).total_seconds() / 60.0 - false_hours * 60.0, 0.0)
        
        self.buffers[act].append((X.copy(), y))
        self.min_et[act] = min(self.min_et[act], y)
        self.max_et[act] = max(self.max_et[act], y)

        err   = abs(y - y_hat)
        self.err_window[act].append(err)
        win_mae = sum(self.err_window[act]) / len(self.err_window[act])
        self.detectors[act].update(err)

        error_flag = False
        if len(self.err_window[act]) == self.err_window[act].maxlen and win_mae > 240:
            error_flag = True
        if self.detectors[act].drift_detected or error_flag:
            self.rebuilding_time += 1
            if self.detectors[act].drift_detected:
                # print(f"Drift")
                width = min(int(self.detectors[act].width), len(self.err_window[act]))
                recent = list(self.buffers[act])[-width:]
                self.detectors[act] = ADWIN(min_window_length=10, delta=0.05)
            else:
                # print(f"ERROR")
                recent = [self.buffers[act][-1]]
            self.err_window[act].clear()
            # print(f"Rebuliding Execution Time Module for act: {act} at {start_ts}")
            new_model = tree.HoeffdingAdaptiveTreeRegressor(
                max_depth=5, leaf_prediction="mean", grace_period=self.grace_period, seed=72)
            for Xb, yb in recent:
                new_model.learn_one(Xb, yb)
            self.excution_time_distrib[act] = new_model
        else:
            self.excution_time_distrib[act].learn_one(X, y)

        self.min_et[act] = min(self.min_et[act], y)
        self.max_et[act] = max(self.max_et[act], y)
        
        

