import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque, defaultdict
import random

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

from river.drift import ADWIN
from sklearn.tree import DecisionTreeClassifier

from modules.simulator import CASE_ID_KEY, ACTIVITY_KEY, END_TIME_KEY

def return_true_alignments(log: pd.DataFrame, net: PetriNet, initial_marking: Marking, final_marking: Marking):
    re_log = log.rename(columns={CASE_ID_KEY: 'case:concept:name',
                                ACTIVITY_KEY: 'concept:name',
                                END_TIME_KEY: 'time:timestamp'}).sort_values(['case:concept:name', 'time:timestamp'])
    params = {"ret_tuple_as_trans_desc": True, alignments.Parameters.SHOW_PROGRESS_BAR: False}
    alignments_ = alignments.apply_log(re_log, net, initial_marking, final_marking, parameters=params)
    groups = list(re_log.groupby('case:concept:name'))
    
    trace_aligned_prefixs = []
    complete_steps = []
    pre_steps = []
    for (case_id, trace_df), ali in zip(groups, alignments_):
        steps = ali['alignment']   # ((log_label, model_label), cost)
        log_len = len(trace_df)
        is_completed = bool(trace_df['is_completed'].iloc[0])
        if log_len == 0:
            continue

        idx_last_log = -1
        for idx, (log_pair, model_pair) in enumerate(steps):
            log_mv = log_pair[0]
            if log_mv != '>>':
                idx_last_log = idx
        if idx_last_log == -1:
            continue

        if is_completed:
            prefix_steps = steps
            complete_steps.append(prefix_steps)
        else:
            prefix_steps = steps[:idx_last_log + 1]
            pre_steps.append(prefix_steps)

        trace_aligned_prefix = [mv_pair for (mv_pair, _) in prefix_steps if mv_pair[1] != '>>']
        trace_aligned_prefixs.append(trace_aligned_prefix)

    return trace_aligned_prefixs


def compute_transition_weights_from_model(models_t: dict, dict_x: dict) -> dict:
    transition_weights = dict()
    list_transitions = list(models_t.keys())
    for t in list_transitions:
        if isinstance(models_t[t], DecisionTreeClassifier):
            # feat_names = sorted(dict_x.keys())
            x_vec = [[dict_x.get(f, 0.0) for f in dict_x.keys()]]
            transition_weights[t] = models_t[t].predict_proba(x_vec)[0][1]
        elif models_t[t]:
            transition_weights[t] = models_t[t]
        else: # no sample
            transition_weights[t] = 1
        
    return transition_weights

def return_fired_transition(transition_weights: dict, enabled_transitions: set) -> PetriNet.Transition:

    weights = [transition_weights[t] for t in enabled_transitions]
    if sum(weights)>0:
        chosen = random.choices(list(enabled_transitions), weights=weights, k=1)[0]
    else:
        chosen = random.choice(list(enabled_transitions))

    return chosen


def update_markings(tkns: list, t_fired: PetriNet.Transition) -> list:

    list_in_arcs = list(t_fired.in_arcs)
    list_out_arcs = list(t_fired.out_arcs)
    for a_in in list_in_arcs:
        tkns.remove(a_in.source)
    for a_out in list_out_arcs:
        tkns.extend([a_out.target])        
    
    return tkns

def return_enabled_and_fired_transitions(
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking, 
        trace_aligned: list,
        neg_num: int=3,
    ) -> tuple:

    visited_transitions = []
    is_fired = []
    tkns = list(initial_marking)
    enabled_transitions = return_enabled_transitions(net, tkns)
    for t_fired_name in trace_aligned:
        for t in net.transitions:
            if t.name == t_fired_name[1]:
                t_fired = t
                break
        # enable but not fired transitions
        not_fired_transitions = list(enabled_transitions-{t_fired})
        if neg_num > 0:
            k = min(neg_num, len(not_fired_transitions))
            not_fired_transitions = random.sample(not_fired_transitions, k)
        for t_not_fired in not_fired_transitions:
            visited_transitions.append(t_not_fired)
            is_fired.append(0)
        # fired transitions
        visited_transitions.append(t_fired)
        is_fired.append(1)
        tkns = update_markings(tkns, t_fired)
        if set(tkns) == set(final_marking):
            return visited_transitions, is_fired
        enabled_transitions = return_enabled_transitions(net, tkns)

    return visited_transitions, is_fired

def return_enabled_transitions(net: PetriNet, tkns: list) -> set:
    
    enabled_t = set()
    list_transitions = list(net.transitions)
    for t in list_transitions:
        if {a.source for a in t.in_arcs}.issubset(tkns):
            enabled_t.add(t)
    
    return enabled_t

def build_training_datasets(
        log: pd.DataFrame, 
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking,
        k_last: int=0,
        neg_num: int=0) -> dict:

    net_transition_labels = list(set([t.label for t in net.transitions if t.label]))
    net_transition_labels = sorted(net_transition_labels)
    def _feature_dict():
        d = {t_l: [] for t_l in net_transition_labels} # cont
        for p in range(1, k_last + 1):
            for t_l in net_transition_labels:
                d[f'last{p}=={t_l}'] = []               # last-k
        d['class'] = []
        return d
    t_dicts_dataset = {t: _feature_dict() for t in net.transitions}
    aligned_traces = return_true_alignments(log, net, initial_marking, final_marking)    
    
    for trace_aligned in aligned_traces:
        visited_transitions, is_fired = return_enabled_and_fired_transitions(net, initial_marking, final_marking, trace_aligned, neg_num)
        for j in range(len(visited_transitions)):
            t = visited_transitions[j]
            t_fired = is_fired[j]
            t_dicts_dataset[t]['class'].append(t_fired)
            # act freq history on current trace
            transitions_fired = [label for label, value in zip(visited_transitions[:j], is_fired[:j]) if value == 1]
            for t_ in net_transition_labels:
                t_dicts_dataset[t][t_].append([x.label for x in transitions_fired].count(t_))
            
            visibel_transitions_fired = [fired_t.label for fired_t in transitions_fired if fired_t.label]
            for p in range(1, k_last + 1):
                lbl_p = visibel_transitions_fired[-p] if len(visibel_transitions_fired) >= p else None
                for t_l in net_transition_labels:
                    t_dicts_dataset[t][f'last{p}=={t_l}'].append(1 if (lbl_p is not None and lbl_p == t_l) else 0)

    datasets_t = {t: pd.DataFrame(t_dicts_dataset[t]) for t in net.transitions}

    return datasets_t

class ProcessModelModule:

    def __init__(self, initial_log: pd.DataFrame, completed_caseids: list):
        print("Process Model Initialization...")
        self.min_pos_neg = 1
        self.k_last = 0
        self.neg_num = 0
        self.t_buffers = defaultdict(lambda: deque(maxlen=1000))    # t -> deque[(X_batch, y_batch)]
        initial_log_complete = initial_log[initial_log[CASE_ID_KEY].isin(completed_caseids)]
        self.net, self.im, self.fm = pm4py.discover_petri_net_inductive(initial_log_complete,
                                                              case_id_key=CASE_ID_KEY,
                                                              activity_key=ACTIVITY_KEY,
                                                              timestamp_key=END_TIME_KEY)
        net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))
        self.transition_labels = sorted(net_transition_labels)
        initial_log = initial_log.copy()
        initial_log["is_completed"] = initial_log[CASE_ID_KEY].isin(completed_caseids)
        self.transition_dist = self.discover_transition_dist(initial_log)
        self.t_detectors = {t: ADWIN(min_window_length=100, delta=0.1) for t in self.net.transitions} 
        self.err_window = {t: deque(maxlen=100) for t in self.net.transitions}
        self.complete_traces = initial_log_complete
        self.net_update_time = 0
        self.dt_new_build_time = 0
        self.dt_update_time = 0
    
    def discover_transition_dist(self, log: pd.DataFrame, max_depth: int = 3) -> dict :
        datasets_t = build_training_datasets(log, self.net, self.im, self.fm, self.k_last, self.neg_num)

        models_t = dict()

        for t in tqdm(self.net.transitions):
            data_t = datasets_t[t]
            if data_t.empty:
                self.t_buffers[t].clear()
                models_t[t] = None
                continue
            
            X_batch = [row.drop('class').to_dict() for _, row in data_t.iterrows()]
            y_batch = data_t['class'].tolist()

            self.t_buffers[t].append((X_batch, y_batch))

            
            ys = y_batch
            if ys.count(1) >= self.min_pos_neg and ys.count(0) >= self.min_pos_neg:
                models_t[t] = DecisionTreeClassifier(random_state=72, max_depth=max_depth)
                X = data_t.drop(columns=['class'])
                y = data_t['class']
                models_t[t].fit(X.values, y.values)
                
            else:
                models_t[t] = None
            

        return models_t
    
    def get_next_transition(self, state: Marking, executed_activities: list):
        net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))
        net_transition_labels = sorted(net_transition_labels)
        dict_x = {t_l: 0 for t_l in net_transition_labels}
        for p in range(1, self.k_last + 1):
            for t_l in net_transition_labels:
                dict_x[f'last{p}=={t_l}'] = 0
        for p in range (1, len(executed_activities)+1):
            act = executed_activities[-p]
            if p <= self.k_last:
                dict_x[f'last{p}=={act}'] = 1 # last-k
            dict_x[act] += 1  # act fre

        enabled_transitions = return_enabled_transitions(self.net, state)
        
        transition_weights = compute_transition_weights_from_model(self.transition_dist, dict_x)
        
        t_fired = return_fired_transition(transition_weights, enabled_transitions)

        state = update_markings(state, t_fired)

        return t_fired, state
    
    def continue_learning(self, trace:pd.DataFrame, error_threshold: float = 0.8, max_depth: int = 3):
        net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))
        net_transition_labels = sorted(net_transition_labels)
        X_row = {t_l: 0 for t_l in net_transition_labels}
        prefix = trace.iloc[:-1].copy()
        for _, event in prefix.iterrows():
            if event[ACTIVITY_KEY] in net_transition_labels:
                X_row[event[ACTIVITY_KEY]] += 1
        for p in range(1, self.k_last + 1):
            lp = prefix.iloc[-p][ACTIVITY_KEY] if len(prefix) >= p else None
            for l in net_transition_labels:
                X_row[f'last{p}=={l}'] = 1 if (lp == l) else 0

        aligned_trace = return_true_alignments(trace, self.net, self.im, self.fm)
        if not aligned_trace:
            return
        aligned_trace = aligned_trace[0]
        
        visited_transitions, is_fired = return_enabled_and_fired_transitions(self.net, self.im, self.fm, aligned_trace, self.neg_num)
        

        j_vis = [i for i,(t,f) in enumerate(zip(visited_transitions, is_fired)) if f==1 and t.label]

        j_prev = j_vis[-2] if len(j_vis)>1 else -1 
        
        batch_X = {t: [] for t in self.net.transitions}
        batch_y = {t: [] for t in self.net.transitions}

        for j in range(j_prev + 1, len(visited_transitions)):
            t = visited_transitions[j]
            y = is_fired[j]
            # the same X_row
            batch_X[t].append(X_row.copy())
            batch_y[t].append(y)

        for t in self.net.transitions:
            X_b = batch_X[t]
            y_b = batch_y[t]
            if not X_b:
                continue

            self.t_buffers[t].append((X_b, y_b))

            clf = self.transition_dist.get(t)
            if not isinstance(clf, DecisionTreeClassifier):
                buf = list(self.t_buffers[t])
                all_X = [x for X_batch, _ in buf for x in X_batch]
                all_y = [y for _, y_batch in buf for y in y_batch]
                ys = all_y
                if ys.count(1) >= self.min_pos_neg and ys.count(0) >= self.min_pos_neg: # sample enough
                    # new desicion tree built
                    # print(f"Build Decision Tree new transition: {t} at {trace.iloc[-1][START_TIME_KEY]}")
                    self.dt_new_build_time += 1
                    new_clf = DecisionTreeClassifier(random_state=72, max_depth=max_depth)
                    X_buf = pd.DataFrame(all_X)
                    y_buf = pd.Series(all_y, name="class")
                    new_clf.fit(X_buf.values, y_buf.values)
                    self.transition_dist[t] = new_clf
                else: # sample not enough
                    self.transition_dist[t] = None
                    
            else:
                # Drift detector
                X_new = pd.DataFrame(X_b)
                y_new = y_b

                classes = list(clf.classes_)
                if len(classes) == 1:
                    if classes[0] == 1:
                        proba = np.ones(len(X_new))
                    else:
                        proba = np.zeros(len(X_new))
                else:
                    proba = clf.predict_proba(X_new.values)[:, 1]

                errors = [abs(y_true - p_hat) for y_true, p_hat in zip(y_new, proba)]
                batch_mae = sum(errors) / len(errors)
                
                self.t_detectors[t].update(batch_mae)
                self.err_window[t].append(batch_mae)

                win_mae = sum(self.err_window[t]) / len(self.err_window[t])

                error_flag = False
                if len(self.err_window[t]) == self.err_window[t].maxlen and win_mae > error_threshold:
                    error_flag = True

                if (self.t_detectors[t].drift_detected or error_flag):
                    if self.t_detectors[t].drift_detected:
                        # print("Drift")
                        width = min(len(self.t_buffers[t]), int(self.t_detectors[t].width))
                        buf = list(self.t_buffers[t])[-width:]
                    else:
                        # print("Error")
                        buf = list(self.t_buffers[t])[-self.err_window[t].maxlen:]

                    all_X = [x for X_batch, _ in buf for x in X_batch]
                    all_y = [y for _, y_batch in buf for y in y_batch]

                    ys = all_y
                    if ys.count(1) >= self.min_pos_neg and ys.count(0) >= self.min_pos_neg:
                        self.dt_update_time += 1
                        # print(f"Decision Tree re-built for transition: {t} at {trace.iloc[-1][START_TIME_KEY]}")
                        new_clf = DecisionTreeClassifier(random_state=72, max_depth=max_depth)

                        X_df_all = pd.DataFrame(all_X)
                        y_ser_all = pd.Series(all_y, name="class")
                        new_clf.fit(X_df_all.values, y_ser_all.values)
                        self.transition_dist[t] = new_clf
                        self.t_detectors[t] = ADWIN(min_window_length=100, delta=0.01)
                
                    self.err_window[t].clear()
    
    def update(self, complete_trace: pd.DataFrame, fitness_threshold: float = 0.8):
        self.complete_traces = pd.concat([self.complete_traces,complete_trace], ignore_index=True)
        df = complete_trace.rename(columns={
            CASE_ID_KEY: 'case:concept:name',
            ACTIVITY_KEY: 'concept:name',
            END_TIME_KEY: 'time:timestamp'
        })
        diag = replay_fitness.apply(df, self.net, self.im, self.fm,
                            variant=replay_fitness.Variants.ALIGNMENT_BASED)
        trace_fitness = float(diag["average_trace_fitness"])   # ∈[0,1]

        if trace_fitness < fitness_threshold:
            # print(f"Update Process Model at {complete_trace.iloc[-1][END_TIME_KEY]}")
            self.net_update_time += 1
            log = self.complete_traces.copy()
            self.net, self.im, self.fm = pm4py.discover_petri_net_inductive(log,
                                                              case_id_key=CASE_ID_KEY,
                                                              activity_key=ACTIVITY_KEY,
                                                              timestamp_key=END_TIME_KEY)
            self.t_buffers.clear()
            net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))
            self.transition_labels = sorted(net_transition_labels)
            log['is_completed'] = True
            self.transition_dist = self.discover_transition_dist(log)
            self.t_detectors = {t: ADWIN(min_window_length=10, delta=0.01) for t in self.net.transitions} 
            self.err_window = {t: deque(maxlen=20) for t in self.net.transitions}


        



