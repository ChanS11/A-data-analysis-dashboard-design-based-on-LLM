import pandas as pd

def filter_data_by_date(df, start_date, end_date):
    if df is None or df.empty: return df
    df_filtered = df.copy()
    df_filtered['IntervalStart'] = pd.to_datetime(df_filtered['IntervalStart'])
    if start_date:
        df_filtered = df_filtered[df_filtered['IntervalStart'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['IntervalStart'] <= pd.to_datetime(end_date)]
    return df_filtered

def recursive_round(obj):
    if isinstance(obj, float): return round(obj, 2)
    elif isinstance(obj, dict): return {k: recursive_round(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [recursive_round(x) for x in obj]
    return obj
