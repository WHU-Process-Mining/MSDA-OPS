import sys
sys.path.append('/home/inspur/zhengchao/MD-OBPS')
import os
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from modules.simulator import Simulator, CASE_ID_KEY, ACTIVITY_KEY, RESOURCE_KEY, START_TIME_KEY, END_TIME_KEY
from utils.evaluation import evaluate_simulation

def run_single_simulation(N,
                          dataset_name,
                          initial_log,
                          initial_completed_caseids,
                          online_stream,
                          simulation_case_num,
                          parameters,
                          test_stream):
    print(f"[PID {os.getpid()}] Simulation {N}...")

    simulator = Simulator(initial_log, initial_completed_caseids, grace_period=25000)

    sim_log = simulator.apply_online_simulation(online_stream, simulation_case_num, parameters)
    sim_path = f"results/{dataset_name}/simulation_log_{N}.csv"
    sim_log.to_csv(sim_path, index=False)

    evaluation_measurement = evaluate_simulation(test_stream, sim_log, N)
    return evaluation_measurement

if __name__ == "__main__":

    # dataset_name = "BPIC2012_W"
    # dataset_name = "BPIC2017_W"
    dataset_name = "ACR"
    # dataset_name = "Production"

    split_ratio = 0.1
    sim_num = 10

    path = f'configs/{dataset_name}.yaml'
    with open(path) as f:
        parameters: dict = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(f'results', exist_ok= True)
    if dataset_name not in os.listdir("results"):
        os.mkdir(f"results/{dataset_name}/")
        

    event_stream = pd.read_csv(parameters["data_path"], dtype=str)
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

    result_list = []
    max_workers = min(sim_num, mp.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for N in range(1, sim_num+1):
            fut = executor.submit(
                run_single_simulation,
                N,
                dataset_name,
                initial_log,
                initial_completed_caseids,
                online_stream,
                simulation_case_num,
                parameters,
                test_stream
            )
            futures.append(fut)

        for fut in futures:
            eval_df = fut.result()
            result_list.append(eval_df)

    result_df = pd.concat(result_list, ignore_index=True)
    
    stats_df = result_df.groupby("metric", sort=False)["distance"].agg(["mean", "std"]).reset_index()
    stats_df["distance"] = stats_df.apply(
            lambda row: f"{round(row['mean'], 5)} ({round(row['std'], 5)})", axis=1
        )
    stats_df["run_num"] = "avg"
    stats_df = stats_df[["run_num", "metric", "distance"]]
    result = pd.concat([result_df, stats_df], ignore_index=True)
    result.to_csv(f"results/{dataset_name}/evaluation.csv", index=False)

        
