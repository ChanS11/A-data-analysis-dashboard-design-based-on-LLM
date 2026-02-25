import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import filter_data_by_date

def plot_stop_analysis(df_stop, company_name, start_date=None, end_date=None, exclude_reasons=None, granularity='Month', exclude_shifts=None):
    """
    Create interactive visualization of stop reason trends over time.
    """
    if df_stop is None or df_stop.empty:
        print(f"No stop data available for {company_name}")
        return

    # Filter Data
    df = filter_data_by_date(df_stop, start_date, end_date)
    
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]

    # Filter out excluded reasons
    if exclude_reasons:
        df = df[~df['StopReason'].isin(exclude_reasons)]

    # Determine grouping period
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
        granularity = 'Month'

    # Get all unique stop reasons and their total stop hours
    all_reasons = df.groupby('StopReason')['TotalStopDuration'].sum().sort_values(ascending=False)
    
    # Determine which reasons to plot (Top 5 of remaining)
    plot_reasons = all_reasons.head(5).index.tolist()
    
    # Filter data to selected reasons
    df_filtered = df[df['StopReason'].isin(plot_reasons)].copy()
    
    if df_filtered.empty:
        print("No data found for the selected criteria.")
        return

    # Group by Period and stop reason, sum stop hours
    trends = df_filtered.groupby(['Period', 'StopReason'])['TotalStopDuration'].sum() / 3600
    
    # Pivot data
    data_pivot = trends.unstack(fill_value=0).sort_index()
    valid_plot_reasons = [r for r in plot_reasons if r in data_pivot.columns]
    data_pivot = data_pivot[valid_plot_reasons]
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'{company_name}: Stop Reason Trends Over {granularity}s', fontsize=16, fontweight='bold')
    
    # Plot 1: Line plot
    ax = axes[0]
    colors = plt.cm.tab10(range(len(valid_plot_reasons)))
    
    for i, reason in enumerate(valid_plot_reasons):
        reason_data = data_pivot[reason]
        ax.plot(range(len(reason_data)), reason_data.values, 
               marker='o', linewidth=2.5, markersize=8, label=reason, 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel(granularity, fontsize=11, fontweight='bold')
    ax.set_ylabel('Stop Hours', fontsize=11, fontweight='bold')
    ax.set_title('Individual Stop Reason Trends (Line Chart)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # X-axis labels
    period_labels = data_pivot.index.astype(str)
    if len(period_labels) > 20:
        step = len(period_labels) // 20 + 1
        ax.set_xticks(range(0, len(period_labels), step))
        ax.set_xticklabels(period_labels[::step], rotation=45, ha='right')
    else:
        ax.set_xticks(range(len(period_labels)))
        ax.set_xticklabels(period_labels, rotation=45, ha='right')
    
    # Plot 2: Stacked area chart
    ax = axes[1]
    ax.stackplot(range(len(data_pivot)), 
                *[data_pivot[col].values for col in data_pivot.columns],
                labels=data_pivot.columns,
                colors=colors[:len(data_pivot.columns)],
                alpha=0.7)
    
    ax.set_xlabel(granularity, fontsize=11, fontweight='bold')
    ax.set_ylabel('Stop Hours', fontsize=11, fontweight='bold')
    ax.set_title('Combined Stop Reason Trends (Stacked Area Chart)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # X-axis labels
    if len(period_labels) > 20:
        step = len(period_labels) // 20 + 1
        ax.set_xticks(range(0, len(period_labels), step))
        ax.set_xticklabels(period_labels[::step], rotation=45, ha='right')
    else:
        ax.set_xticks(range(len(period_labels)))
        ax.set_xticklabels(period_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_operator_comment_analysis(df_stop, company_name, start_date=None, end_date=None, exclude_shifts=None):
    """
    Analyze operator comments to find common issues.
    """
    if df_stop is None or df_stop.empty:
        print(f"No stop data available for {company_name}")
        return
    if 'Comment' not in df_stop.columns:
        print(f"No 'Comment' column found in stop data for {company_name}")
        return

    df = filter_data_by_date(df_stop, start_date, end_date)
    if exclude_shifts and 'Shift' in df.columns:
        df = df[~df['Shift'].isin(exclude_shifts)]
        
    df_comments = df[df['Comment'].notna() & (df['Comment'].str.strip() != '')].copy()
    if df_comments.empty:
        print("No operator comments found in the selected period.")
        return
        
    total_stops = len(df)
    total_comments = len(df_comments)
    comment_rate = (total_comments / total_stops * 100) if total_stops > 0 else 0
    
    reason_counts = df_comments['StopReason'].value_counts().head(10)
    reason_intensity = []
    for reason in reason_counts.index:
        total_for_reason = len(df[df['StopReason'] == reason])
        comments_for_reason = len(df_comments[df_comments['StopReason'] == reason])
        intensity = (comments_for_reason / total_for_reason * 100) if total_for_reason > 0 else 0
        reason_intensity.append(intensity)
        
    # Count full recurring comments instead of individual words
    comment_counts = df_comments['Comment'].str.strip().value_counts().head(10)
    
    df_comments['CommentLength'] = df_comments['Comment'].str.len()
    longest_comments = df_comments.sort_values('CommentLength', ascending=False).head(5)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{company_name}: Operator Comment Analysis', fontsize=16, fontweight='bold')
    
    ax1 = axes[0]
    y_pos = np.arange(len(reason_counts))
    ax1.barh(y_pos, reason_counts.values, color='#4ECDC4', alpha=0.8)
    for i, (count, intensity) in enumerate(zip(reason_counts.values, reason_intensity)):
        ax1.text(count + 0.5, i, f"{count} ({intensity:.1f}% of stops)", va='center')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(reason_counts.index)
    ax1.invert_yaxis()
    ax1.set_title('Top Stop Reasons with Comments', fontweight='bold')
    
    ax2 = axes[1]
    if not comment_counts.empty:
        y_pos_c = np.arange(len(comment_counts))
        # Truncate long comments for display
        labels = [c[:40] + '...' if len(c) > 40 else c for c in comment_counts.index]
        
        ax2.barh(y_pos_c, comment_counts.values, color='#FF6B6B', alpha=0.8)
        ax2.set_yticks(y_pos_c)
        ax2.set_yticklabels(labels)
        ax2.invert_yaxis()
        ax2.set_title('Most Frequent Recurring Comments', fontweight='bold')
        
        # Add count labels
        for i, v in enumerate(comment_counts.values):
            ax2.text(v + 0.1, i, str(v), va='center')
    else:
        ax2.text(0.5, 0.5, 'No recurring comments found', ha='center')

    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*80}")
    print(f"OPERATOR COMMENT INSIGHTS - {company_name}")
    print(f"{'='*80}")
    print(f"Total Stops in Period: {total_stops}")
    print(f"Stops with Comments: {total_comments} ({comment_rate:.1f}%)")
    print(f"\n--- TOP 5 LONGEST COMMENTS (Potential Complex Issues) ---")
    for idx, row in longest_comments.iterrows():
        print(f"\n[Reason: {row['StopReason']} | Duration: {row['TotalStopDuration']/60:.1f} min]")
        print(f"\"{row['Comment']}\"")
