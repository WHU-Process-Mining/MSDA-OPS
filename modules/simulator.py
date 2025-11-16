CASE_ID_KEY = 'Case ID'
ACTIVITY_KEY = 'Activity'
RESOURCE_KEY = 'Resource'
START_TIME_KEY = 'Start Timestamp'
END_TIME_KEY = 'Complete Timestamp'

import pandas as pd
from collections import Counter
from tqdm import tqdm

from modules.arrival_time_module import ArrivalTimeModule
from modules.resource_module import ResourceModule
from modules.process_model_module import ProcessModelModule
from modules.excution_time_module import ExcutionTimeModule
from modules.waiting_time_module import WaitingTimeModule
from utils.time_utils import add_minutes_with_calendar





class Simulator:

    def __init__(
            self,
            inital_log: pd.DataFrame,
            completed_caseids: list,
            grace_period: int = 1000,
        ):  

        self.arrival_model = ArrivalTimeModule(inital_log, grace_period)
        self.resource_model = ResourceModule(inital_log)
        self.process_model = ProcessModelModule(inital_log, completed_caseids, grace_period)
        self.excution_time_model = ExcutionTimeModule(inital_log, self.resource_model.resource_calendar, grace_period)
        self.waiting_time_model = WaitingTimeModule(inital_log, self.resource_model.resource_calendar, grace_period)
        ongoing_case_ids = set(inital_log[CASE_ID_KEY].unique())-set(completed_caseids)
        self.ongoing_trace = dict()
        for case_id in ongoing_case_ids:
            self.ongoing_trace[case_id] = (inital_log[inital_log[CASE_ID_KEY] == case_id]
                                            .sort_values([START_TIME_KEY, END_TIME_KEY], kind="mergesort")
                                            .reset_index(drop=True))        
        self.sim_traces = []
        self.case_id = 1
        self.processed_events = 0
        self.current_timestamp = None
    
    def apply_online_simulation(self, streams: pd.DataFrame, sim_num: int):
        self.sim_num = sim_num
        counts = Counter(streams[CASE_ID_KEY])
        for _, event in tqdm(streams.iterrows(), total=len(streams)):
            case_id = event[CASE_ID_KEY]
            counts[case_id] -= 1

            if case_id in self.ongoing_trace.keys():
                self.ongoing_trace[case_id].loc[len(self.ongoing_trace[case_id])] = event

                if counts[case_id] == 0: # trace is complete
                    complete_trace = self.ongoing_trace.pop(case_id)
                    self.process_model.update(complete_trace)
            else: # new trace
                self.ongoing_trace[case_id] = pd.DataFrame([event])
                initial_trace_time = self.arrival_model.get_arrival_time(event)
                act_trace = self._generate_activity_trace()
                trace = self._generate_traces(act_trace, initial_trace_time, )
                self.sim_traces.extend(trace)
                self.case_id += 1

            if case_id in self.ongoing_trace:
                update_trace = self.ongoing_trace[case_id]
            else:# complete just now
                update_trace = complete_trace
            
            self.update_event_level_model(update_trace)

        print(f'Arrival Model Updating Num:{self.arrival_model.update_num}')
        print(f'Process Model Updating Num:{self.process_model.net_update_time}')
        print(f'Process Decision Tree New-build Num:{self.process_model.dt_new_build_time}')
        print(f'Process Decision Tree Re-build Num:{self.process_model.dt_update_time}')
        print(f'Resource Calendar Updating Num:{self.resource_model.calendar_rebuilt_time}')
        print(f'Activty-Resource Distribution Updating Num:{self.resource_model.act_res_dist_rebuilt_time}')
        print(f'Waiting Time Model Updating Num:{self.waiting_time_model.rebuilding_time}')
        print(f'Execution Time Model Updating Num:{self.excution_time_model.rebuilding_time}')

        sim_log = pd.DataFrame(self.sim_traces)
        assert self.sim_num == sim_log[CASE_ID_KEY].nunique(), "Simulation num is Wrong!"
        return sim_log
    
    def update_event_level_model(self, trace: pd.DataFrame):
        cur_event = trace.iloc[-1]
        self.process_model.continue_learning(trace)
        self.resource_model.update_model(cur_event)
        self.excution_time_model.update(trace, self.resource_model.resource_calendar)
        self.waiting_time_model.update(trace, self.resource_model.resource_calendar)
    

    def _generate_activity_trace(self,):
        """ Generate Activity Traces """

        activity_trace = []

        while len(activity_trace)==0:
            tkns = list(self.process_model.im)
            t_fired, tkns = self.process_model.get_next_transition(tkns, activity_trace)
            if t_fired.label:
                activity_trace.append(t_fired.label)

            while set(tkns) != set(self.process_model.fm):
                t_fired, tkns = self.process_model.get_next_transition(tkns, activity_trace)
                if t_fired.label:
                    activity_trace.append(t_fired.label)
        
        return activity_trace
    
    def _generate_traces(self, activity_trace, start_sim_time):
        trace = []
        current_ts = start_sim_time
        for i, act in enumerate(activity_trace):
            resource = self.resource_model.get_resource(act)

            if i == 0: # first event
                start_ts = start_sim_time
            else:
                n_active = 0
                for pre_e in self.sim_traces:
                    if pre_e[RESOURCE_KEY] == resource and pre_e[START_TIME_KEY] <= current_ts < pre_e[END_TIME_KEY]:
                        n_active += 1

                for evt in trace:
                    if evt[RESOURCE_KEY] == resource and evt[END_TIME_KEY] > current_ts and evt[START_TIME_KEY] < current_ts:
                        n_active += 1
            
                waiting_time = self.waiting_time_model.get_waiting_time(resource, current_ts, n_active)
                start_ts = add_minutes_with_calendar(current_ts, waiting_time, self.resource_model.resource_calendar[resource])

            
            ex_time = self.excution_time_model.get_execution_time(act, resource, start_ts, activity_trace[:i])
            end_ts = add_minutes_with_calendar(start_ts, ex_time, self.resource_model.resource_calendar[resource])
                
            current_ts = end_ts

            trace.append({CASE_ID_KEY: f'case_{self.case_id}', 
                           ACTIVITY_KEY: act, 
                           START_TIME_KEY: start_ts, 
                           END_TIME_KEY: end_ts, 
                           RESOURCE_KEY: resource})
        
        return trace
            


