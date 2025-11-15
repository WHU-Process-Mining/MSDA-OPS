import pandas as pd
from tqdm import tqdm
import random
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.tokenreplay.algorithm import apply as token_replay
from collections import deque, defaultdict
from river import tree, stats
from river.drift import ADWIN
from modules.simulator import CASE_ID_KEY, ACTIVITY_KEY, START_TIME_KEY, END_TIME_KEY

def compute_transition_weights_from_model(models_t: dict, dict_x: dict) -> dict:
    transition_weights = dict()
    list_transitions = list(models_t.keys())
    for t in list_transitions:
        if models_t[t] is None:
            transition_weights[t] = 0.01
        else:
            transition_weights[t] = models_t[t].predict_proba_one(dict_x)[1]
        
    return transition_weights

def return_transitions_frequency(
        log: pd.DataFrame, 
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking
    ) -> dict:

    alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]

    frequency_t = {t: 0 for t in net.transitions}
    list_transitions = list(net.transitions)
    for trace in aligned_traces:
        for align in trace:
            name_t = align[1]
            for t in list_transitions:
                if t.name == name_t:
                    frequency_t[t] += 1
                    break

    return frequency_t


def return_fired_transition(transition_weights: dict, enabled_transitions: set) -> PetriNet.Transition:

    weights = [float(transition_weights.get(t, 0.01)) for t in enabled_transitions]
    chosen = random.choices(list(enabled_transitions), weights=weights, k=1)[0]
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
        trace_aligned: list
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
        final_marking: Marking) -> dict:

    net_transition_labels = list(set([t.label for t in net.transitions if t.label]))
    t_dicts_dataset = {t: {t_l: [] for t_l in net_transition_labels} |
                          {'class': []} for t in net.transitions}

    re_log = log.rename(columns={'Case ID': 'case:concept:name',
                                'Activity': 'concept:name',
                                'Complete Timestamp': 'time:timestamp'})
    alignments_ = alignments.apply_log(re_log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    # delete model move
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]
    
    for trace_aligned in aligned_traces:
        visited_transitions, is_fired = return_enabled_and_fired_transitions(net, initial_marking, final_marking, trace_aligned)
        for j in range(len(visited_transitions)):
            t = visited_transitions[j]
            t_fired = is_fired[j]
            t_dicts_dataset[t]['class'].append(t_fired)
            # act freq history on current trace
            transitions_fired = [label for label, value in zip(visited_transitions[:j], is_fired[:j]) if value == 1]
            for t_ in net_transition_labels:
                t_dicts_dataset[t][t_].append([x.label for x in transitions_fired].count(t_))

    datasets_t = {t: pd.DataFrame(t_dicts_dataset[t]) for t in net.transitions}

    return datasets_t

class ProcessModelModule:

    def __init__(self, initial_log: pd.DataFrame, completed_caseids: list, grace_period: int = 1000):
        print("Process Model Initialization...")
        self.grace_period = grace_period
        initial_log_complete = initial_log[initial_log[CASE_ID_KEY].isin(completed_caseids)]
        self.net, self.im, self.fm = pm4py.discover_petri_net_inductive(initial_log_complete,
                                                              case_id_key=CASE_ID_KEY,
                                                              activity_key=ACTIVITY_KEY,
                                                              timestamp_key=END_TIME_KEY)
        self.transition_labels= list(set([t.label for t in self.net.transitions if t.label]))
        self.transition_dist = self.discover_transition_dist(initial_log, grace_period)

        self.t_buffers = defaultdict(lambda: deque(maxlen=500))    # t -> deque[(X, y)]
        self.t_detectors = defaultdict(lambda: ADWIN(min_window_length=200, delta=0.01))  # 概念漂移
        self.t_loss_ewm = defaultdict(lambda: stats.EWMean(0.95))  # 平滑损失
        self.min_pos_neg = 1

        self.complete_traces = []
        self.net_update_time = 0
        self.dt_update_time = 0
        self.detector = ADWIN(min_window_length=50, delta=0.01)
    
    def discover_transition_dist(self, log: pd.DataFrame, grace_period: int = 1000, max_depth: int = 3) -> dict :
        datasets_t = build_training_datasets(log, self.net, self.im, self.fm)

        models_t = dict()

        for t in tqdm(self.net.transitions):
            data_t = datasets_t[t]
            if len(data_t['class'].unique())<2:
                models_t[t] = None
                continue
            
            models_t[t] = tree.HoeffdingAdaptiveTreeClassifier(seed=72, leaf_prediction="mc", max_depth=max_depth, grace_period=grace_period)

            for _, row in data_t.iterrows():
                X_row = row.drop('class').to_dict()
                y_row = row['class']
                models_t[t].learn_one(X_row, y_row)

        return models_t
    
    def get_next_transition(self, state: Marking, dict_x: dict= None):
        if not dict_x:
            dict_x = {t_l: 0 for t_l in self.transition_labels}

        enabled_transitions = return_enabled_transitions(self.net, state)
        
        transition_weights = compute_transition_weights_from_model(self.transition_dist, dict_x)
        
        t_fired = return_fired_transition(transition_weights, enabled_transitions)

        state = update_markings(state, t_fired)

        return t_fired, state, dict_x
    
    def continue_learning(self, trace:pd.DataFrame, max_depth: int = 3):
        net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))
        
        X_row = {t_l: 0 for t_l in net_transition_labels}

        re_trace = trace.rename(columns={CASE_ID_KEY: 'case:concept:name',
                                ACTIVITY_KEY: 'concept:name',
                                END_TIME_KEY: 'time:timestamp'})
        
        alignments_ = alignments.apply_log(re_trace, self.net, self.im, self.fm, parameters={"ret_tuple_as_trans_desc": True})
        # delete model move
        aligned_trace = [y[0] for y in alignments_[0]['alignment'] if y[0][1]!='>>'] 
        
        
        visited_transitions, is_fired = return_enabled_and_fired_transitions(self.net, self.im, self.fm, aligned_trace)
        
        t_target = trace.iloc[-1][ACTIVITY_KEY]
        cand_curr = [i for i,(t,f) in enumerate(zip(visited_transitions, is_fired)) if f==1 and t.label==t_target]
        if not cand_curr: # new act
            return
        j_curr = cand_curr[-1]
        prev_vis = [i for i,(t,f) in enumerate(zip(visited_transitions, is_fired)) if i<j_curr and f==1 and t.label]
        j_prev = prev_vis[-1] if prev_vis else -1

        start_idx = j_prev + 1
        end_idx   = j_curr + 1

        for i in range(start_idx):
            t = visited_transitions[i]
            if t.label:
                X_row[t.label] += 1
        
        t_dicts_dataset = {t: {'feature':[], 'class': []} for t in self.net.transitions}
        
        for j in range(start_idx, end_idx):
            t = visited_transitions[j]
            t_fired = is_fired[j]
            t_dicts_dataset[t]['feature'].append(X_row.copy())
            t_dicts_dataset[t]['class'].append(t_fired)
            if t.label:
                X_row[t.label] += 1
        
        for t, dict_t in t_dicts_dataset.items():
            for X, y in zip(dict_t['feature'], dict_t['class']):
                self.t_buffers[t].append((X, y))

            clf = self.transition_dist.get(t)
            if clf is None:
                # new desicion tree built
                print(f"Build Decision Tree new transition: {t} at {trace.iloc[-1][START_TIME_KEY]}")
                self.dt_update_time += 1
                ys = [y for _, y in self.t_buffers[t]]
                if ys.count(1) >= self.min_pos_neg and ys.count(0) >= self.min_pos_neg: # sample enough
                    clf = tree.HoeffdingAdaptiveTreeClassifier(
                        seed=72, leaf_prediction="mc", max_depth=max_depth, grace_period=self.grace_period
                    )
                    for Xb, yb in self.t_buffers[t]:
                        clf.learn_one(Xb, yb)
                    self.transition_dist[t] = clf
                else: # sample not enough
                    continue # 
            else:
                # continue learning
                for X, y in zip(dict_t['feature'], dict_t['class']):
                    proba = clf.predict_proba_one(X).get(1, 0.0) if hasattr(clf, "predict_proba_one") else float(clf.predict_one(X) == 1)
                    loss  = abs(y - proba)  # 简单Brier-like
                    self.t_loss_ewm[t].update(loss)
                    scale = max(self.t_loss_ewm[t].get(), 1e-6)
                    x = min(loss / scale, 10.0) / 10.0
                    self.t_detectors[t].update(x)

                    if self.t_detectors[t].drift_detected:
                        self.dt_update_time += 1
                        print(f"Decision Tree re-built for transition: {t} at {trace.iloc[-1][START_TIME_KEY]}")
                        new_clf = tree.HoeffdingAdaptiveTreeClassifier(
                            seed=72, leaf_prediction="mc", max_depth=max_depth, grace_period=self.grace_period
                        )
                        for Xb, yb in self.t_buffers[t]:
                            new_clf.learn_one(Xb, yb)
                        self.transition_dist[t] = new_clf
                        self.t_detectors[t] = ADWIN(min_window_length=200, delta=0.01)
                    else:
                        clf.learn_one(X, y)

    def update(self, complete_trace):
        self.complete_traces.append(complete_trace)
        df = complete_trace.rename(columns={
            CASE_ID_KEY: 'case:concept:name',
            ACTIVITY_KEY: 'concept:name',
            END_TIME_KEY: 'time:timestamp'
        })
        diag = replay_fitness.apply(df, self.net, self.im, self.fm,
                            variant=replay_fitness.Variants.ALIGNMENT_BASED)
        trace_fitness = float(diag["average_trace_fitness"])   # ∈[0,1]
        self.detector.update(1.0 - trace_fitness) 

        if self.detector.drift_detected and len(self.complete_traces)>50:
            print(f"Update Process Model at {complete_trace.iloc[-1][END_TIME_KEY]}")
            self.net_update_time += 1
            
            win = min(len(self.complete_traces), 200)
            recent_traces = self.complete_traces[-win:]
            log = pd.concat(recent_traces, ignore_index=True)
            self.net, self.im, self.fm = pm4py.discover_petri_net_inductive(log,
                                                              case_id_key=CASE_ID_KEY,
                                                              activity_key=ACTIVITY_KEY,
                                                              timestamp_key=END_TIME_KEY)
            self.transition_labels = list({t.label for t in self.net.transitions if t.label})
            self.transition_dist = self.discover_transition_dist(log, self.grace_period)

            self.complete_traces = []
            self.detector = ADWIN(min_window_length=200, delta=0.01)

        



