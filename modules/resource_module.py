import pandas as pd
import random
import math
from collections import Counter, deque
from river.drift import ADWIN
from modules.simulator import ACTIVITY_KEY, RESOURCE_KEY, START_TIME_KEY, END_TIME_KEY

def normalize_dist(cnt: Counter) -> dict:
    laplace = 1.0
    # 拉普拉斯平滑 + 归一化
    keys = list(cnt.keys())
    Z = sum(cnt.values()) + laplace * len(keys)
    if Z <= 0:
        return {}
    return {k: (cnt[k] + laplace) / Z for k in keys}

def is_on_duty(calendar, res, ts) -> int:
    res_map = calendar.get(res)
    if not res_map:
        return 0
    day_map = res_map.get(ts.weekday())
    return int(day_map.get(ts.hour))

class ResourceModule:

    def __init__(self, initial_log):
        print("Resource Module Initialization")
        self.resources = list(initial_log[RESOURCE_KEY].unique())
        self.resource_calendar = self.discover_res_calendars(initial_log)
        self.act_resource_dist = self.discover_act_resource_dist(initial_log)
        activities = initial_log[ACTIVITY_KEY].unique()
        self.act_buffers = {act: deque(maxlen=100) for act in activities}
        self.act_detectors = {act: ADWIN(min_window_length=10, delta=0.05) for act in activities} # 活动分配资源的检测
        self.act_res_counts = {act: Counter() for act in activities}
        self.err_window = {act: deque(maxlen=10) for act in activities}
        for act, dist in self.act_resource_dist.items():
            for r, p in dist.items():
                self.act_res_counts[act][r] += max(p, 1e-4)*100
        
        self.calendar_rebuilt_time = 0
        self.act_res_dist_rebuilt_time = 0

    def _default_calendar(self):
        return {wd: {h: True for h in range(24)} for wd in range(7)}
    
    def discover_res_calendars(self, log: pd.DataFrame, thr_h: float = 0.0, thr_wd: float = 0.0) -> dict:
        N_events_per_hour_res = {res: {wd: {h: 0 for h in range(24)} for wd in range(7)} for res in self.resources}

        for _, event in log.iterrows():
            res = event[RESOURCE_KEY]
            ts = event[START_TIME_KEY]
            N_events_per_hour_res[res][ts.weekday()][ts.hour] += 1
            ts = event[END_TIME_KEY]
            N_events_per_hour_res[res][ts.weekday()][ts.hour] += 1

        N_events_per_hour_res_perc = {res: {wd: {h: 0 for h in range(24)} for wd in range(7)} for res in self.resources}

        for res in self.resources:
            for weekday in range(7):
                for h in range(24):
                    if sum(N_events_per_hour_res[res][weekday].values()):
                        N_events_per_hour_res_perc[res][weekday][h] = N_events_per_hour_res[res][weekday][h] / sum(N_events_per_hour_res[res][weekday].values())
                    else:
                        N_events_per_hour_res_perc[res][weekday][h] = 0


        N_events_per_wd_res = {res: {wd: 0 for wd in range(7)} for res in self.resources}

        for res in self.resources:
            for weekday in range(7):
                N_events_per_wd_res[res][weekday] += sum(N_events_per_hour_res[res][weekday].values())


        N_events_per_wd_res_perc = {res: {wd: N_events_per_wd_res[res][wd]/sum(N_events_per_wd_res[res].values()) if sum(N_events_per_wd_res[res].values()) else 0 for wd in range(7)} for res in self.resources}

        calendar_wd_hour_res = {res: {wd: {h: False for h in range(24)} for wd in range(7)} for res in self.resources}

        for res in self.resources:
            for wd in range(7):
                if N_events_per_wd_res_perc[res][wd] <= thr_wd:
                    continue
                else:
                    for hour in range(24):
                        calendar_wd_hour_res[res][wd][hour] = N_events_per_hour_res_perc[res][wd][hour] > thr_h

        return calendar_wd_hour_res
    
    def discover_act_resource_dist(self, log: pd.DataFrame) -> dict:
        act_freq = log[ACTIVITY_KEY].value_counts().to_dict()
        act_resources = dict()

        for _, event in log.iterrows():
            res = event[RESOURCE_KEY]
            act = event[ACTIVITY_KEY]
            if act not in act_resources.keys():
                act_resources[act] = dict()
            if res not in act_resources[act].keys():
                act_resources[act][res] = 1
            else:
                act_resources[act][res] += 1

        for act in act_resources.keys():
            for res in act_resources[act].keys():
                act_resources[act][res] = act_resources[act][res]/act_freq[act]

        return act_resources
    
    def get_resource(self, activity):
        dist = self.act_resource_dist.get(activity)

        if not dist: # activity first appear
            cand = self.resources
            w    = [1.0 / len(cand)] * len(cand)
        else:
            cand = list(dist.keys())
            w    = list(dist.values())
        resource = random.choices(cand, weights=w)[0]
        return resource


    def update_model(self, event:pd.Series, error_threshold: float = 0.9):

        act = event[ACTIVITY_KEY]
        res = event[RESOURCE_KEY]
        start_ts = event[START_TIME_KEY]
        end_ts = event[END_TIME_KEY]

        if act not in self.act_detectors:
            self.act_detectors[act] = ADWIN(min_window_length=10, delta=0.05)
        if act not in self.act_res_counts:
            self.act_res_counts[act] = Counter()
        if act not in self.act_buffers:
            self.act_buffers[act] = deque(maxlen=100)
        if act not in self.err_window:
            self.err_window[act] = deque(maxlen=10)

        if res not in self.resource_calendar:
            self.resource_calendar[res] = self._default_calendar()
        if res not in self.resources:
            self.resources.append(res)
        
        # calendar update
        for t in (start_ts, end_ts):
            in_duty = is_on_duty(self.resource_calendar, res, t)
            if not in_duty:
                self.calendar_rebuilt_time += 1
                self.resource_calendar[res][t.weekday()][t.hour]=True

        # act-> res distribution
        dist_a = self.act_resource_dist.get(act, {})
        p = dist_a.get(res, 1e-6)
        s = -math.log(p)
        x = min(s / 14.0, 1.0) # surprise at 1
        self.act_detectors[act].update(x)
        self.act_buffers[act].append(res)

        self.err_window[act].append(x)
        win_supr = sum(self.err_window[act]) / len(self.err_window[act])

        error_flag = False
        if len(self.err_window[act]) == self.err_window[act].maxlen and win_supr > error_threshold:
            error_flag = True

        if self.act_detectors[act].drift_detected or error_flag:
            self.act_res_dist_rebuilt_time += 1
            # print(f"Activity Resource Rebuilt for act: {act} at {start_ts}")
            if self.act_detectors[act].drift_detected:
                # print("Drift")
                width = min(int(self.act_detectors[act].width), len(self.act_buffers[act]))
                recent = list(self.act_buffers[act])[-width:]
                self.act_detectors[act] = ADWIN(min_window_length=10, delta=0.05)
            else:
                # print("Error")
                recent = list(self.act_buffers[act])[-self.err_window[act].maxlen:]
            self.err_window[act].clear()
            cnt = Counter(recent)
            self.act_res_counts[act] = cnt
            self.act_resource_dist[act] = normalize_dist(cnt)
        else:
            # decay the count
            for k in list(self.act_res_counts[act].keys()):
                self.act_res_counts[act][k] *= 0.995
                if self.act_res_counts[act][k] < 1e-8:
                    del self.act_res_counts[act][k]
            self.act_res_counts[act][res] += 1.0
            self.act_resource_dist[act] = normalize_dist(self.act_res_counts[act])