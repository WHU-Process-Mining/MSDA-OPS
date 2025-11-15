

import datetime
from enum import Enum
import pandas as pd
from log_distance_measures.absolute_event_distribution import (
    absolute_event_distribution_distance,
    discretize_to_hour,
)
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import (
    circadian_event_distribution_distance,
)
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, EventLogIDs
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import (
    cycle_time_distribution_distance,
)
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance

from modules.simulator import CASE_ID_KEY, ACTIVITY_KEY, RESOURCE_KEY, START_TIME_KEY, END_TIME_KEY

# Define metrics to compute
class Metric(str, Enum):
    """
    Enum class storing the metrics used to evaluate the quality of a BPS model.

    Attributes
    ----------
    DL : str
        Control-flow Log Distance metric based in the Damerau-Levenshtein distance.
    TWO_GRAM_DISTANCE : str
        Two-gram distance metric.
    THREE_GRAM_DISTANCE : str
        Three-gram distance metric.
    CIRCADIAN_EMD : str
        Earth Mover's Distance (EMD) for circadian event distribution.
    CIRCADIAN_WORKFORCE_EMD : str
        EMD for circadian workforce distribution.
    ARRIVAL_EMD : str
        EMD for arrival event distribution.
    RELATIVE_EMD : str
        EMD for relative event distribution.
    ABSOLUTE_EMD : str
        EMD for absolute event distribution.
    CYCLE_TIME_EMD : str
        EMD for cycle time distribution.
    """

    DL = "dl"
    TWO_GRAM_DISTANCE = "two_gram_distance"
    THREE_GRAM_DISTANCE = "three_gram_distance"
    CIRCADIAN_EMD = "circadian_event_distribution"
    CIRCADIAN_WORKFORCE_EMD = "circadian_workforce_distribution"
    ARRIVAL_EMD = "arrival_event_distribution"
    RELATIVE_EMD = "relative_event_distribution"
    ABSOLUTE_EMD = "absolute_event_distribution"
    CYCLE_TIME_EMD = "cycle_time_distribution"

def compute_metric(
    metric: Metric,
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """Computes the distance between an original (test) event log and a simulated one.

    :param metric: The metric to compute.
    :param original_log: Original event log.
    :param original_log_ids: Column names of the original event log.
    :param simulated_log: Simulated event log.
    :param simulated_log_ids: Column names of the simulated event log.

    :return: The computed metric.
    """

    if metric is Metric.DL:
        result = get_dl(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.TWO_GRAM_DISTANCE:
        result = get_n_grams_distribution_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, 2)
    elif metric is Metric.THREE_GRAM_DISTANCE:
        result = get_n_grams_distribution_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, 3)
    elif metric is Metric.CIRCADIAN_EMD:
        result = get_circadian_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.CIRCADIAN_WORKFORCE_EMD:
        result = get_circadian_workforce_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.ARRIVAL_EMD:
        result = get_arrival_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.RELATIVE_EMD:
        result = get_relative_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.ABSOLUTE_EMD:
        result = get_absolute_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    elif metric is Metric.CYCLE_TIME_EMD:
        result = get_cycle_time_emd(original_log, original_log_ids, simulated_log, simulated_log_ids)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return result


def get_absolute_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the timestamps of two event logs are, discretizing
    the timestamps by absolute hour.
    """

    emd = absolute_event_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        AbsoluteTimestampType.BOTH,
        discretize_to_hour,
    )
    return emd


def get_cycle_time_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the cycle time discretized histograms of two event logs are.
    """
    emd = cycle_time_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        datetime.timedelta(hours=1),
    )
    return emd


def get_circadian_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the timestamps of two event logs are, comparing all
    the instants recorded in the same weekday together (e.g., Monday), and discretizing them to the hour in the day.
    """
    emd = circadian_event_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        AbsoluteTimestampType.BOTH,
    )
    return emd


def get_circadian_workforce_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the active resources of two event logs are, comparing the
    average number of active resources recorded each weekday at each hour (e.g., Monday 10am).
    """
    emd = circadian_workforce_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
    )
    return emd


def get_arrival_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the histograms of the case arrivals of two event logs are.
    """
    emd = case_arrival_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
    )
    return emd


def get_relative_emd(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    """
    Distance measure computing how different the distribution of the events with each case (i.e., relative to their
    start) of two event logs are.
    """
    emd = relative_event_distribution_distance(
        original_log,
        original_log_ids,
        simulated_log,
        simulated_log_ids,
        AbsoluteTimestampType.BOTH,
    )
    return emd


def get_n_grams_distribution_distance(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
    n: int = 3,
) -> float:
    """
    Distance measure between two event logs computing the difference in the frequencies of the n-grams observed in
    the event logs (being the n-grams of an event log all the groups of n consecutive elements observed in it).
    :return: The MAE between the frequency of trigrams occurring in one log vs the other.
    """
    mae = n_gram_distribution_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, n=n)
    return mae


def get_dl(
    original_log: pd.DataFrame,
    original_log_ids: EventLogIDs,
    simulated_log: pd.DataFrame,
    simulated_log_ids: EventLogIDs,
) -> float:
    cfld = control_flow_log_distance(original_log, original_log_ids, simulated_log, simulated_log_ids, True)
    return cfld

def time_parse(log, time_columns):
    for col in time_columns:
        log[col] = pd.to_datetime(log[col], utc=True, format='mixed')
    return log

def evaluate_simulation(test_log, sim_log, rep):
    """
    Evaluate the simulated log against the validation log using various distance metrics.

    Parameters:
        dataset_name (str): The name of the dataset.
        run_name (str): The name of the simulation run.
        rep (int): The repetition number for the simulation run.

    Returns:
        measurements (list): A list of dictionaries containing the metric results.
    """
    
    measurements = []

    log_ids = EventLogIDs(
        case=CASE_ID_KEY,
        activity=ACTIVITY_KEY,
        resource=RESOURCE_KEY,
        start_time=START_TIME_KEY,
        end_time= END_TIME_KEY,
    )
    
    assert test_log[CASE_ID_KEY].nunique() == sim_log[CASE_ID_KEY].nunique(), "Number of cases in test and simulated logs do not match."
    # assert test_log['Start Timestamp'].min() == simulated_log['start_timestamp'].min(), "Start timestamps do not match."
    
    for metric in Metric:
        metric_name = metric.value
        value = compute_metric(metric, test_log, log_ids, sim_log, log_ids)
        measurements.append({"run_num": rep, "metric": metric_name, "distance": value})
        print(f"Run {rep}, Metric: {metric_name}, Distance: {value}")
    measurements_df = pd.DataFrame(measurements)
    return measurements_df

# if __name__ == "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")

#     dataset_name = "BPIC2017_W"
#     output_path = f"results/{dataset_name}/online/results.csv"
#     rep = 10
    
#     measurements_df = paraller_execute_evaluation(dataset_name, rep)
#     # Compute mean and std for each metric
#     stats_df = measurements_df.groupby("metric", sort=False)["distance"].agg(["mean", "std"]).reset_index()
#     stats_df["distance"] = stats_df.apply(
#             lambda row: f"{round(row['mean'], 5)} ({round(row['std'], 5)})", axis=1
#         )
#     stats_df["run_num"] = "avg"
#     stats_df = stats_df[["run_num", "metric", "distance"]]
#     result = pd.concat([measurements_df, stats_df], ignore_index=True)
#     result.to_csv(output_path, index=False)
    