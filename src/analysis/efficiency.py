import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import filter_data_by_date

def plot_efficiency_trends(df_worktime, company_name, start_date=None, end_date=None, exclude_shifts=None, granularity='Month'):
    """
    Create interactive visualization of production efficiency trends.
    """
    if df_worktime is None or df_worktime.empty:
        print(f"No worktime data available for {company_name}")
        return

    df = filter_data_by_date(df_worktime, start_date, end_date)

    # Filter shifts
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]

    # Grouping
    xlabel = granularity
    if granularity == 'Year':
        df['Period'] = df['IntervalStart'].dt.to_period('Y')
    elif granularity == 'Month':
        df['Period'] = df['IntervalStart'].dt.to_period('M')
    elif granularity == 'Week':
        df['Period'] = df['IntervalStart'].dt.to_period('W')
    elif granularity == 'Day':
        df['Period'] = df['IntervalStart'].dt.to_period('D')
    else:
        df['Period'] = df['IntervalStart'].dt.to_period('M')
        xlabel = 'Month'

    # Aggregation
    grouped = df.groupby('Period')[['ScheduledDuration', 'ProductionTimeDuration', 'MicroStopDuration', 'ReworkedEffectiveTime']].sum()
    grouped = grouped / 3600 # Convert to hours
    
    # Calculate Efficiency %
    grouped['Efficiency'] = (grouped['ProductionTimeDuration'] / grouped['ScheduledDuration'] * 100).fillna(0)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Bar chart for Hours
    periods = grouped.index.astype(str)
    x = range(len(periods))
    
    ax1.bar(x, grouped['ScheduledDuration'], label='Scheduled Time', alpha=0.3, color='gray')
    ax1.bar(x, grouped['ProductionTimeDuration'], label='Production Time', alpha=0.6, color='green')
    
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hours', fontsize=12, fontweight='bold')
    ax1.set_title(f'{company_name}: Production Efficiency Trends ({xlabel})', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Line chart for Efficiency %
    ax2 = ax1.twinx()
    ax2.plot(x, grouped['Efficiency'], color='blue', marker='o', linewidth=2, label='Efficiency %')
    ax2.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 110)
    
    # Reference line
    ax2.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Target (85%)')
    ax2.legend(loc='upper right')
    
    # X-axis labels
    if len(periods) > 20:
        step = len(periods) // 20 + 1
        ax1.set_xticks(range(0, len(periods), step))
        ax1.set_xticklabels(periods[::step], rotation=45, ha='right')
    else:
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods, rotation=45, ha='right')
        
    plt.tight_layout()
    plt.show()
    
    # Summary Stats
    print(f"\n{'='*80}")
    print(f"EFFICIENCY ANALYSIS SUMMARY - {company_name}")
    print(f"{'='*80}")
    
    total_sched = grouped['ScheduledDuration'].sum()
    total_prod = grouped['ProductionTimeDuration'].sum()
    overall_eff = (total_prod / total_sched * 100) if total_sched > 0 else 0
    
    print(f"Overall Efficiency:     {overall_eff:.2f}%")
    print(f"Total Scheduled Hours:  {total_sched:,.1f}")
    print(f"Total Production Hours: {total_prod:,.1f}")
    print(f"Total Microstop Hours:  {grouped['MicroStopDuration'].sum():,.1f}")
    print(f"Total Rework Hours:     {grouped['ReworkedEffectiveTime'].sum():,.1f}")

def plot_machine_activity(df_worktime, company_name, start_date=None, end_date=None, exclude_shifts=None):
    """
    Calculate and visualize normalized production activity by individual machine.
    """
    if df_worktime is None or df_worktime.empty:
        print(f"No worktime data available for {company_name}")
        return

    df = filter_data_by_date(df_worktime, start_date, end_date)
    
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]
    
    df['Date'] = df['IntervalStart'].dt.date
    df['ProductionTimeDuration'] = df['ProductionTimeDuration'].fillna(0)
    df['ScheduledDuration'] = df['ScheduledDuration'].fillna(0)
    
    df['ActivityPercentage'] = 0.0
    scheduled_mask = df['ScheduledDuration'] > 0
    df.loc[scheduled_mask, 'ActivityPercentage'] = (df.loc[scheduled_mask, 'ProductionTimeDuration'] / 
        df.loc[scheduled_mask, 'ScheduledDuration'] * 100
    )
    
    daily_machine_activity = df.groupby(['Date', 'MeasurePoint']).agg({
        'ActivityPercentage': 'mean',
        'ScheduledDuration': 'sum'
    }).reset_index()
    
    machine_groups = df.groupby('MeasurePoint')
    machine_activity_list = []
    for machine, group in machine_groups:
        avg_activity = group['ActivityPercentage'].mean()
        total_scheduled = group['ScheduledDuration'].sum()
        machine_activity_list.append({
            'MeasurePoint': machine,
            'AvgActivityPercentage': avg_activity,
            'TotalScheduledDuration': total_scheduled
        })
    
    machine_activity = pd.DataFrame(machine_activity_list)
    machine_activity = machine_activity.sort_values('AvgActivityPercentage', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'{company_name}: Machine Activity Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    machine_colors = plt.cm.Set3(np.linspace(0, 1, len(machine_activity)))
    bars = ax.barh(machine_activity['MeasurePoint'], machine_activity['AvgActivityPercentage'], color=machine_colors)
    ax.set_title('Average Activity % (Production / Scheduled)', fontweight='bold')
    ax.set_xlabel('Activity %')
    ax.set_xlim([0, 100])
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax = axes[1]
    machines = sorted(daily_machine_activity['MeasurePoint'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(machines)))
    
    for i, machine in enumerate(machines):
        machine_data = daily_machine_activity[daily_machine_activity['MeasurePoint'] == machine]
        ax.plot(machine_data['Date'], machine_data['ActivityPercentage'], 
                label=machine, marker='o', linewidth=1.5, color=colors[i],
                markersize=4, alpha=0.8)
    
    ax.set_title('Daily Activity Trend by Machine', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Activity %')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.show()
