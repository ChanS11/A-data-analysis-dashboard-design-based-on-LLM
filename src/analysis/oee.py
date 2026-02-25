import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import filter_data_by_date

def get_calculation_settings(df_setting, company_name):
    if df_setting is None or df_setting.empty:
        return {'MicrostopAsPerformanceLoss': False, 'CalculatePerformanceFromWeightedProducedUnits': False, 'ReworkAsQualityLoss': False}
    settings = {}
    for setting_name in ['MicrostopAsPerformanceLoss', 'CalculatePerformanceFromWeightedProducedUnits', 'ReworkAsQualityLoss']:
        matching_rows = df_setting[df_setting.columns[df_setting.columns.str.lower() == setting_name.lower()]]
        if not matching_rows.empty:
            value = matching_rows.iloc[0, 0]
            settings[setting_name] = bool(value) if value is not None else False
        else:
            settings[setting_name] = False
    return settings

def calculate_oee_data(df_worktime, df_setting, company_name, start_date=None, end_date=None, exclude_shifts=None):
    if df_worktime is None or df_worktime.empty: return None
    settings = get_calculation_settings(df_setting, company_name)
    
    df = filter_data_by_date(df_worktime, start_date, end_date)
    
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]
        
    df['Date'] = df['IntervalStart'].dt.date
    duration_cols = ['ScheduledDuration', 'StopDuration', 'SetupStopDuration', 'NoWorkTimeStopDuration', 
                     'MicroStopDuration', 'ExcludedDuration', 'ProductionTimeDuration', 'ReworkedEffectiveTime',
                     'UsedEffectiveTime', 'ScrappedEffectiveTime', 'ReworkedUnits', 'ScrappedUnits',
                     'OptimalProducedUnits', 'OptimalProducedUnitsNoMicroStop', 'ProducedUnits']
    for col in duration_cols:
        if col in df.columns: df[col] = df[col].fillna(0)
        else: df[col] = 0
            
    # Availability
    df['Availability'] = 0.0
    avail_denom = df['ScheduledDuration'] - df['ExcludedDuration']
    avail_mask = avail_denom > 0
    avail_num = (df['ScheduledDuration'] - df['StopDuration'] - df['SetupStopDuration'] - df['NoWorkTimeStopDuration'])
    if settings['MicrostopAsPerformanceLoss']: avail_num += df['MicroStopDuration']
    df.loc[avail_mask, 'Availability'] = (avail_num[avail_mask] / avail_denom[avail_mask] * 100)
    df.loc[~avail_mask, 'Availability'] = 100.0
    df['Availability'] = df['Availability'].clip(0, 100)
    
    # Performance
    df['Performance'] = 100.0
    if settings['MicrostopAsPerformanceLoss'] and settings['CalculatePerformanceFromWeightedProducedUnits']:
        perf_denom = df['ProductionTimeDuration'] + df['MicroStopDuration']
        perf_mask = perf_denom > 0
        perf_num = df['UsedEffectiveTime'].copy()
        if not settings['ReworkAsQualityLoss']: perf_num -= df['ReworkedEffectiveTime']
        df.loc[perf_mask, 'Performance'] = (perf_num[perf_mask] / perf_denom[perf_mask] * 100)
    elif settings['MicrostopAsPerformanceLoss']:
        perf_mask = df['OptimalProducedUnitsNoMicroStop'] > 0
        perf_num = df['ProducedUnits'].copy()
        if not settings['ReworkAsQualityLoss']: perf_num -= df['ReworkedUnits']
        cond = df['OptimalProducedUnitsNoMicroStop'] > df['ProducedUnits']
        df.loc[perf_mask & cond, 'Performance'] = (perf_num[perf_mask & cond] / df.loc[perf_mask & cond, 'OptimalProducedUnitsNoMicroStop'] * 100)
    elif settings['CalculatePerformanceFromWeightedProducedUnits']:
        perf_mask = df['ProductionTimeDuration'] > 0
        perf_num = df['UsedEffectiveTime'].copy()
        if not settings['ReworkAsQualityLoss']: perf_num -= df['ReworkedEffectiveTime']
        df.loc[perf_mask, 'Performance'] = (perf_num[perf_mask] / df.loc[perf_mask, 'ProductionTimeDuration'] * 100)
    else:
        perf_mask = df['OptimalProducedUnits'] > 0
        perf_num = df['ProducedUnits'].copy()
        if not settings['ReworkAsQualityLoss']: perf_num -= df['ReworkedUnits']
        cond = df['OptimalProducedUnits'] > df['ProducedUnits']
        df.loc[perf_mask & cond, 'Performance'] = (perf_num[perf_mask & cond] / df.loc[perf_mask & cond, 'OptimalProducedUnits'] * 100)
    df['Performance'] = df['Performance'].clip(0, 100)
    
    # Quality
    df['Quality'] = 100.0
    if settings['CalculatePerformanceFromWeightedProducedUnits']:
        qual_num = df['UsedEffectiveTime'] - df['ScrappedEffectiveTime']
        if settings['ReworkAsQualityLoss']: qual_num -= df['ReworkedEffectiveTime']
        qual_check = qual_num >= 0
        qual_mask = (df['UsedEffectiveTime'] > 0) & qual_check
        df.loc[qual_mask, 'Quality'] = (qual_num[qual_mask] / df.loc[qual_mask, 'UsedEffectiveTime'] * 100)
        df.loc[~qual_check, 'Quality'] = 0.0
    else:
        qual_num = df['ProducedUnits'] - df['ScrappedUnits']
        if settings['ReworkAsQualityLoss']: qual_num -= df['ReworkedUnits']
        qual_check = qual_num >= 0
        qual_mask = (df['ProducedUnits'] > 0) & qual_check
        df.loc[qual_mask, 'Quality'] = (qual_num[qual_mask] / df.loc[qual_mask, 'ProducedUnits'] * 100)
        df.loc[~qual_check, 'Quality'] = 0.0
    df['Quality'] = df['Quality'].clip(0, 100)
    
    # OEE
    df['OEE'] = (df['Availability'] / 100) * (df['Performance'] / 100) * (df['Quality'] / 100) * 100
    df['OEE'] = df['OEE'].clip(0, 100)
    
    return df

def plot_oee_analysis(df_worktime, df_setting, company_name, start_date=None, end_date=None, exclude_shifts=None):
    df = calculate_oee_data(df_worktime, df_setting, company_name, start_date, end_date, exclude_shifts)
    if df is None:
        print(f"No worktime data available for {company_name}")
        return

    daily_oee = df.groupby('Date').agg({'Availability': 'mean', 'Performance': 'mean', 'Quality': 'mean', 'OEE': 'mean'}).reset_index()
    machine_oee = df.groupby('MeasurePoint').agg({'Availability': 'mean', 'Performance': 'mean', 'Quality': 'mean', 'OEE': 'mean'}).reset_index().sort_values('OEE', ascending=False)
    overall = {'Availability': df['Availability'].mean(), 'Performance': df['Performance'].mean(), 'Quality': df['Quality'].mean(), 'OEE': df['OEE'].mean()}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{company_name}: OEE Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    metrics = ['Availability', 'Performance', 'Quality', 'OEE']
    values = [overall[m] for m in metrics]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#2C3E50']
    bars = ax.bar(metrics, values, color=colors, alpha=0.8)
    ax.set_title('Overall OEE Metrics', fontweight='bold')
    ax.set_ylim(0, 110)
    for bar in bars: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
                
    ax = axes[0, 1]
    ax.plot(daily_oee['Date'], daily_oee['Availability'], label='Availability', color='#FF6B6B', alpha=0.6)
    ax.plot(daily_oee['Date'], daily_oee['Performance'], label='Performance', color='#4ECDC4', alpha=0.6)
    ax.plot(daily_oee['Date'], daily_oee['Quality'], label='Quality', color='#95E1D3', alpha=0.6)
    ax.plot(daily_oee['Date'], daily_oee['OEE'], label='OEE', color='#2C3E50', linewidth=2.5, marker='o')
    ax.set_title('Daily OEE Trend', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    ax = axes[1, 0]
    top_machines = machine_oee.head(10)
    y_pos = np.arange(len(top_machines))
    ax.barh(y_pos, top_machines['OEE'], color='#2C3E50', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_machines['MeasurePoint'])
    ax.invert_yaxis()
    ax.set_title('Top 10 Machines by OEE', fontweight='bold')
    ax.set_xlim(0, 100)
    
    ax = axes[1, 1]
    bot_machines = machine_oee.tail(10)
    y_pos = np.arange(len(bot_machines))
    ax.barh(y_pos, bot_machines['OEE'], color='#FF6B6B', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bot_machines['MeasurePoint'])
    ax.invert_yaxis()
    ax.set_title('Bottom 10 Machines by OEE', fontweight='bold')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()
    print(f"\nOverall OEE: {overall['OEE']:.1f}%")

def plot_shift_analysis(df_worktime, df_setting, company_name, start_date=None, end_date=None, exclude_shifts=None):
    if df_worktime is None or df_worktime.empty:
        print(f"No worktime data available for {company_name}")
        return
    if 'Shift' not in df_worktime.columns:
        print(f"⚠️ Shift column missing in data for {company_name}. Cannot perform shift analysis.")
        return

    df = calculate_oee_data(df_worktime, df_setting, company_name, start_date, end_date, exclude_shifts)
    df = df[df['Shift'].notna()].copy()
    if df.empty:
        print("No data available for the selected period after filtering.")
        return

    shift_metrics = df.groupby('Shift').agg({
        'OEE': 'mean', 'Availability': 'mean', 'Performance': 'mean', 'Quality': 'mean',
        'ProducedUnits': 'sum', 'StopDuration': 'sum'
    }).reset_index()
    shift_metrics['StopHours'] = shift_metrics['StopDuration'] / 3600
    shift_metrics = shift_metrics.sort_values('Shift')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{company_name}: Shift Performance Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    shifts = shift_metrics['Shift'].astype(str)
    x = np.arange(len(shifts))
    width = 0.2
    ax.bar(x - width*1.5, shift_metrics['Availability'], width, label='Availability', color='#FF6B6B', alpha=0.8)
    ax.bar(x - width*0.5, shift_metrics['Performance'], width, label='Performance', color='#4ECDC4', alpha=0.8)
    ax.bar(x + width*0.5, shift_metrics['Quality'], width, label='Quality', color='#95E1D3', alpha=0.8)
    ax.bar(x + width*1.5, shift_metrics['OEE'], width, label='OEE', color='#2C3E50', alpha=0.9)
    ax.set_title('OEE Components by Shift', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shifts)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', ncol=2)
    
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.bar(x - 0.2, shift_metrics['ProducedUnits'], 0.4, label='Produced Units', color='#4ECDC4', alpha=0.7)
    l2 = ax2.bar(x + 0.2, shift_metrics['StopHours'], 0.4, label='Stop Hours', color='#FF6B6B', alpha=0.7)
    ax.set_title('Production Volume vs Stop Time', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shifts)
    ax.legend([l1, l2], ['Produced Units', 'Stop Hours'], loc='upper left')
    
    ax = axes[1, 0]
    daily_shift = df.groupby(['Date', 'Shift'])['OEE'].mean().unstack()
    for col in daily_shift.columns:
        series = daily_shift[col].dropna()
        if not series.empty: ax.plot(series.index, series.values, marker='o', label=f'{col}', linewidth=2, alpha=0.8)
    ax.set_title('Daily OEE Trend by Shift', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    ax = axes[1, 1]
    plot_data = []
    labels = []
    for shift in shift_metrics['Shift']:
        shift_data = df[df['Shift'] == shift]['OEE'].dropna()
        plot_data.append(shift_data.values)
        labels.append(str(shift))
    ax.boxplot(plot_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor='#95E1D3', color='#2C3E50'))
    ax.set_title('OEE Variability Distribution by Shift', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
