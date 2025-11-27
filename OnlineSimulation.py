import sys
sys.path.append('/home/inspur/zhengchao/MD-OBPS')
import os
import pandas as pd

from modules.simulator import Simulator, CASE_ID_KEY, ACTIVITY_KEY, RESOURCE_KEY, START_TIME_KEY, END_TIME_KEY
from utils.evaluation import evaluate_simulation


if __name__ == "__main__":

    # dataset_name = "BPIC2012_W"
    dataset_name = "BPIC2017_W"
    # dataset_name = "ACR"
    # dataset_name = "Production"

    if dataset_name == "BPIC2012_W":
        parameters = {'process_fitness_threshold': 0.5,
                  'process_error_threshold': 0.9,
                  'arrival_error_threshold':60,
                  'res_error_threshold': 0.5,
                  'wt_error_threshold': 240,
                  'et_error_threshold':180,}
    if dataset_name == "BPIC2017_W":
        parameters = {'process_fitness_threshold': 0.7,
                  'process_error_threshold': 0.9,
                  'arrival_error_threshold':300,
                  'res_error_threshold': 0.7,
                  'wt_error_threshold': 180,
                  'et_error_threshold':300,}
    elif dataset_name == "ACR":
        parameters = {'process_fitness_threshold':0.7, 
                  'process_error_threshold':0.5,
                  'arrival_error_threshold':60,
                  'res_error_threshold': 0.5,
                  'wt_error_threshold': 180,
                  'et_error_threshold':300,
                  }
    elif dataset_name == "Production":
        parameters = {'process_fitness_threshold':0.7, 
                  'process_error_threshold':0.5,
                  'arrival_error_threshold':60,
                  'res_error_threshold': 0.5,
                  'wt_error_threshold': 180,
                  'et_error_threshold':300,
                  }

    split_ratio = 0.1
    sim_num = 10
    stream_path = f"/home/inspur/zhengchao/BPS_datasets/{dataset_name}/online-process/event_stream.csv"
    os.makedirs(f'results', exist_ok= True)
    if dataset_name not in os.listdir("results"):
        os.mkdir(f"results/{dataset_name}/")
        

    event_stream = pd.read_csv(stream_path, dtype=str)
    event_stream[CASE_ID_KEY] = event_stream[CASE_ID_KEY].astype(str)
    event_stream[ACTIVITY_KEY] = event_stream[ACTIVITY_KEY].astype(str)
    event_stream[RESOURCE_KEY] = event_stream[RESOURCE_KEY].astype(str)
    event_stream[START_TIME_KEY] = pd.to_datetime(event_stream[START_TIME_KEY], utc=True, format="mixed")
    event_stream[END_TIME_KEY] = pd.to_datetime(event_stream[END_TIME_KEY], utc=True, format="mixed")
    
    split_time = event_stream[START_TIME_KEY].min() + split_ratio * (event_stream[END_TIME_KEY].max() - event_stream[START_TIME_KEY].min())
    
    case_span = (event_stream
             .groupby(CASE_ID_KEY)
             .agg(case_start=(START_TIME_KEY, "min"),
                  case_end=(END_TIME_KEY, "max")))
    # train set all cases that end before separation time.

    initial_log = event_stream[event_stream[START_TIME_KEY] <= split_time]
    initial_completed_caseids = set(case_span.loc[case_span["case_end"] <= split_time].index.astype(str))
    ongoing_caseids = set(initial_log[CASE_ID_KEY].unique()) - initial_completed_caseids

    online_stream = event_stream[event_stream[START_TIME_KEY] > split_time]
    test_stream = online_stream[~online_stream[CASE_ID_KEY].isin(ongoing_caseids)]
    simulation_case_num = test_stream[CASE_ID_KEY].nunique()
    simulation_start_time = test_stream[START_TIME_KEY].min()
    print("Number of cases in the simulation: ", simulation_case_num)
    print("Simulation start time: ", simulation_start_time)

    result_df = pd.DataFrame()
    for N in range(1, sim_num+1):
        print(f"Simulation {N}...")
        simulator = Simulator(initial_log, initial_completed_caseids, grace_period=25000)
        sim_log = simulator.apply_online_simulation(online_stream, simulation_case_num, parameters)
        sim_log.to_csv(f"results/{dataset_name}/simulation_log_{N}.csv", index=False)
        evaluation_measurement = evaluate_simulation(test_stream, sim_log, N)
        result_df = pd.concat([result_df, evaluation_measurement], ignore_index=True)
    
    stats_df = result_df.groupby("metric", sort=False)["distance"].agg(["mean", "std"]).reset_index()
    stats_df["distance"] = stats_df.apply(
            lambda row: f"{round(row['mean'], 5)} ({round(row['std'], 5)})", axis=1
        )
    stats_df["run_num"] = "avg"
    stats_df = stats_df[["run_num", "metric", "distance"]]
    result = pd.concat([result_df, stats_df], ignore_index=True)
    result.to_csv(f"results/{dataset_name}/evaluation.csv", index=False)

        
