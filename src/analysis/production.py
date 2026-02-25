import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import filter_data_by_date

def plot_production_output(df_worktime, company_name, start_date=None, end_date=None, exclude_shifts=None):
    """
    Analyze production output metrics.
    """
    if df_worktime is None or df_worktime.empty:
        print(f"No worktime data available for {company_name}")
        return
    
    df = filter_data_by_date(df_worktime, start_date, end_date)
    
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]
    
    df['Date'] = df['IntervalStart'].dt.date
    for col in ['ProducedUnits', 'ApprovedUnits', 'OptimalProducedUnits']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    df['ApprovalRate'] = 0.0
    produced_mask = df['ProducedUnits'] > 0
    df.loc[produced_mask, 'ApprovalRate'] = (df.loc[produced_mask, 'ApprovedUnits'] / df.loc[produced_mask, 'ProducedUnits'] * 100)
    
    total_produced = df['ProducedUnits'].sum()
    total_approved = df['ApprovedUnits'].sum()
    overall_approval_rate = (total_approved / total_produced * 100) if total_produced > 0 else 0
    
    daily_production = df.groupby('Date').agg({
        'ProducedUnits': 'sum', 'ApprovedUnits': 'sum'
    }).reset_index()
    
    machine_production = df.groupby('MeasurePoint').agg({
        'ProducedUnits': 'sum', 'ApprovedUnits': 'sum'
    }).reset_index()
    machine_production['ApprovalRate'] = (machine_production['ApprovedUnits'] / machine_production['ProducedUnits'] * 100).fillna(0)
    machine_production = machine_production.sort_values('ProducedUnits', ascending=False)
    machine_production = machine_production[machine_production['ProducedUnits'] > 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{company_name}: Production Output & Quality Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(daily_production['Date'], daily_production['ProducedUnits'], marker='o', linewidth=2, label='Produced', color='#4ECDC4')
    ax.plot(daily_production['Date'], daily_production['ApprovedUnits'], marker='s', linewidth=2, label='Approved', color='#95E1D3')
    ax.set_title('Daily Production Trend', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    ax = axes[0, 1]
    reject_rate = 100 - overall_approval_rate
    sizes = [overall_approval_rate, reject_rate]
    labels = [f'Approved\n({overall_approval_rate:.1f}%)', f'Rejected\n({reject_rate:.1f}%)']
    colors = ['#95E1D3', '#FF6B6B']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
    ax.set_title('Overall Quality: Approval vs Reject Rate', fontweight='bold')
    
    ax = axes[1, 0]
    top_machines = machine_production.head(10)
    y_pos = np.arange(len(top_machines))
    ax.barh(y_pos, top_machines['ProducedUnits'], color='#4ECDC4', alpha=0.7, label='Produced')
    ax.barh(y_pos, top_machines['ApprovedUnits'], color='#95E1D3', alpha=0.9, label='Approved')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_machines['MeasurePoint'])
    ax.invert_yaxis()
    ax.set_title('Top 10 Machines by Volume', fontweight='bold')
    ax.legend()
    
    ax = axes[1, 1]
    top_quality = machine_production.sort_values('ApprovalRate', ascending=True).tail(10)
    y_pos = np.arange(len(top_quality))
    bars = ax.barh(y_pos, top_quality['ApprovalRate'], color='#FF6B6B', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_quality['MeasurePoint'])
    ax.set_xlabel('Approval Rate %')
    ax.set_xlim(0, 110)
    ax.set_title('Top 10 Machines by Approval Rate', fontweight='bold')
    for i, v in enumerate(top_quality['ApprovalRate']):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    plt.show()

def plot_scrap_rate_analysis(df_worktime, company_name, start_date=None, end_date=None, exclude_shifts=None, granularity='Month'):
    if df_worktime is None or df_worktime.empty:
        print(f"No worktime data available for {company_name}")
        return

    df = filter_data_by_date(df_worktime, start_date, end_date)
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]
        
    for col in ['ProducedUnits', 'ScrappedUnits']:
        if col in df.columns: df[col] = df[col].fillna(0)
            
    if granularity == 'Year': df['Period'] = df['IntervalStart'].dt.to_period('Y')
    elif granularity == 'Month': df['Period'] = df['IntervalStart'].dt.to_period('M')
    elif granularity == 'Week': df['Period'] = df['IntervalStart'].dt.to_period('W')
    elif granularity == 'Day': df['Period'] = df['IntervalStart'].dt.to_period('D')
    else: df['Period'] = df['IntervalStart'].dt.to_period('M')
        
    trend_data = df.groupby('Period')[['ProducedUnits', 'ScrappedUnits']].sum()
    trend_data['ScrapRate'] = (trend_data['ScrappedUnits'] / trend_data['ProducedUnits'] * 100).fillna(0)
    
    machine_data = df.groupby('MeasurePoint')[['ProducedUnits', 'ScrappedUnits']].sum()
    machine_data['ScrapRate'] = (machine_data['ScrappedUnits'] / machine_data['ProducedUnits'] * 100).fillna(0)
    machine_data = machine_data.sort_values('ScrapRate', ascending=False)
    
    article_data = pd.DataFrame()
    if 'ArticleName' in df.columns:
        article_data = df.groupby('ArticleName')[['ProducedUnits', 'ScrappedUnits']].sum()
        article_data = article_data[article_data['ProducedUnits'] > 100]
        article_data['ScrapRate'] = (article_data['ScrappedUnits'] / article_data['ProducedUnits'] * 100).fillna(0)
        article_data = article_data.sort_values('ScrapRate', ascending=False).head(10)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f'{company_name}: Scrap Rate Analysis', fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(gs[0, :])
    periods = trend_data.index.astype(str)
    ax1.plot(periods, trend_data['ScrapRate'], marker='o', linewidth=2.5, color='#FF6B6B', label='Scrap Rate %')
    ax1.set_title(f'Scrap Rate Trend ({granularity})', fontweight='bold')
    ax1.set_ylabel('Scrap Rate (%)')
    ax1.grid(True, alpha=0.3)
    if len(periods) > 20:
        step = len(periods) // 20 + 1
        ax1.set_xticks(range(0, len(periods), step))
        ax1.set_xticklabels(periods[::step], rotation=45, ha='right')
    else:
        ax1.set_xticks(range(len(periods)))
        ax1.set_xticklabels(periods, rotation=45, ha='right')
        
    ax2 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(machine_data))
    ax2.barh(y_pos, machine_data['ScrapRate'], color='#FF6B6B', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(machine_data.index)
    ax2.invert_yaxis()
    ax2.set_title('Scrap Rate by Machine', fontweight='bold')
    for i, v in enumerate(machine_data['ScrapRate']): ax2.text(v + 0.1, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)
        
    ax3 = fig.add_subplot(gs[1, 1])
    if not article_data.empty:
        y_pos = np.arange(len(article_data))
        ax3.barh(y_pos, article_data['ScrapRate'], color='#C0392B', alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(article_data.index)
        ax3.invert_yaxis()
        ax3.set_title('Top 10 Articles by Scrap Rate (>100 units produced)', fontweight='bold')
        for i, v in enumerate(article_data['ScrapRate']): ax3.text(v + 0.1, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No Article Data Available', ha='center', va='center')
        
    plt.tight_layout()
    plt.show()
    
    total_produced = trend_data['ProducedUnits'].sum()
    total_scrapped = trend_data['ScrappedUnits'].sum()
    avg_scrap_rate = (total_scrapped / total_produced * 100) if total_produced > 0 else 0
    print(f"\nOverall Scrap Rate: {avg_scrap_rate:.2f}%")
    print(f"Total Scrapped Units: {int(total_scrapped):,}")
    print(f"Total Produced Units: {int(total_produced):,}")
