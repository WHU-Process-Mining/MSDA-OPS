from datetime import datetime, timedelta


def count_false_hours(calendar: dict, start_ts: datetime, end_ts: datetime) -> int:
    false_hours_count = 0
    current_time = start_ts
    
    while current_time < end_ts:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if calendar.get(weekday, {}).get(hour) == False:
            false_hours_count += 1
            
        current_time += timedelta(hours=1)

    return false_hours_count

def add_minutes_with_calendar(start_ts: datetime, minutes_to_add: int, calendar: dict) -> datetime:
    remaining_minutes = minutes_to_add
    current_time = start_ts

    while remaining_minutes > 0:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if calendar.get(weekday, {}).get(hour, False):
            minutes_in_current_hour = min(remaining_minutes, 60 - current_time.minute)
            
            current_time += timedelta(minutes=minutes_in_current_hour)
            remaining_minutes -= minutes_in_current_hour
        else:
            current_time = (current_time + timedelta(hours=1)).replace(minute=0)

    return current_time


def cal_error_with_calendar(predicted_ts: datetime, real_ts: datetime, calendar: dict) -> float:
    """
    计算预测时间 predicted_ts 与真实时间 real_ts 之间的误差（单位：分钟），
    但不计入日历中不可用(False)时段的分钟数。

    """
    if predicted_ts == real_ts:
        return 0.0

    # 确保 start < end
    start, end = (predicted_ts, real_ts) if predicted_ts < real_ts else (real_ts, predicted_ts)
    total_working_minutes = 0.0

    current = start.replace(second=0, microsecond=0)
    while current < end:
        # 当前小时的结束时刻
        hour_end = (current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        segment_end = min(hour_end, end)

        weekday = current.weekday()
        hour = current.hour
        is_working = calendar.get(weekday, {}).get(hour, False)

        if is_working:
            mins = (segment_end - current).total_seconds() / 60.0
            total_working_minutes += mins

        current = segment_end

    return total_working_minutes
