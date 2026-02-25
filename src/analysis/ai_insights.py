import pandas as pd
import numpy as np
import json
import os
from openai import OpenAI
from src.utils import filter_data_by_date, recursive_round
from src.analysis.oee import calculate_oee_data
from src.config import OPENAI_API_KEY

def generate_ai_insights(df_worktime, df_stop, df_setting, company_name, start_date=None, end_date=None, exclude_reasons=None, exclude_shifts=None, include_recommendations=True):
    """
    Generate a comprehensive JSON data summary for the LLM to analyze.
    """
    # 1. Prepare Data
    current_start = pd.to_datetime(start_date) if start_date else df_worktime['IntervalStart'].min()
    current_end = pd.to_datetime(end_date) if end_date else df_worktime['IntervalStart'].max()
    
    duration = current_end - current_start
    previous_end = current_start - pd.Timedelta(seconds=1)
    previous_start = previous_end - duration
    
    df_oee_curr = calculate_oee_data(df_worktime, df_setting, company_name, current_start, current_end, exclude_shifts)
    df_oee_prev = calculate_oee_data(df_worktime, df_setting, company_name, previous_start, previous_end, exclude_shifts)
    df_full_oee = calculate_oee_data(df_worktime, df_setting, company_name, exclude_shifts=exclude_shifts)
    
    df_s_curr = filter_data_by_date(df_stop, current_start, current_end)
    df_s_prev = filter_data_by_date(df_stop, previous_start, previous_end)
    
    if exclude_reasons:
        df_s_curr = df_s_curr[~df_s_curr['StopReason'].isin(exclude_reasons)]
        df_s_prev = df_s_prev[~df_s_prev['StopReason'].isin(exclude_reasons)]
        
    if exclude_shifts and 'Shift' in df_s_curr.columns:
        df_s_curr = df_s_curr[~df_s_curr['Shift'].isin(exclude_shifts)]
        df_s_prev = df_s_prev[~df_s_prev['Shift'].isin(exclude_shifts)]

    # --- 2. EXECUTIVE SUMMARY METRICS ---
    def get_metrics(df_oee):
        if df_oee is None or df_oee.empty: return {}
        return {
            'OEE': df_oee['OEE'].mean(),
            'Availability': df_oee['Availability'].mean(),
            'Performance': df_oee['Performance'].mean(),
            'Quality': df_oee['Quality'].mean(),
            'Total_Produced_Units': df_oee['ProducedUnits'].sum() if 'ProducedUnits' in df_oee else 0,
            'Total_Scrapped_Units': df_oee['ScrappedUnits'].sum() if 'ScrappedUnits' in df_oee else 0,
            'Scrap_Rate': (df_oee['ScrappedUnits'].sum() / df_oee['ProducedUnits'].sum() * 100) if 'ProducedUnits' in df_oee and df_oee['ProducedUnits'].sum() > 0 else 0
        }

    current_metrics = get_metrics(df_oee_curr)
    previous_metrics = get_metrics(df_oee_prev)
    
    metrics_change = {}
    for k, v in current_metrics.items():
        prev = previous_metrics.get(k, 0)
        if prev != 0:
            change = ((v - prev) / prev) * 100
            metrics_change[k] = f"{change:+.1f}%"
        else:
            metrics_change[k] = "N/A"

    # --- 3. TRENDS (Aggregated) ---
    trend_summary = {}
    if df_oee_curr is not None and not df_oee_curr.empty:
        daily_series = df_oee_curr.groupby(df_oee_curr['IntervalStart'].dt.date)['OEE'].mean()
        if len(daily_series) > 1:
            x = np.arange(len(daily_series))
            slope, _ = np.polyfit(x, daily_series.values, 1)
            trend_direction = "Improving" if slope > 0.1 else ("Declining" if slope < -0.1 else "Stable")
            volatility = daily_series.std()
            if len(daily_series) > 60:
                agg_series = df_oee_curr.set_index('IntervalStart').resample('M')['OEE'].mean()
                period_type = "Monthly_Averages"
            elif len(daily_series) > 14:
                agg_series = df_oee_curr.set_index('IntervalStart').resample('W')['OEE'].mean()
                period_type = "Weekly_Averages"
            else:
                agg_series = daily_series
                period_type = "Daily_Values"
            trend_summary = {
                "Trend_Direction": trend_direction, "Slope": slope, "Volatility_StdDev": volatility,
                period_type: {str(k.date()) if hasattr(k, 'date') else str(k): v for k, v in agg_series.items()}
            }

    # --- 4. MACHINE INSIGHTS ---
    top_machines = []
    bottom_machines = []
    if df_oee_curr is not None and not df_oee_curr.empty:
        mach_groups = df_oee_curr.groupby('MeasurePoint')
        machine_stats = []
        for name, group in mach_groups:
            prod = group['ProducedUnits'].sum() if 'ProducedUnits' in group else 0
            scrap = group['ScrappedUnits'].sum() if 'ScrappedUnits' in group else 0
            scrap_rate = (scrap / prod * 100) if prod > 0 else 0
            machine_stats.append({
                'Machine': name, 'OEE': group['OEE'].mean(), 'Availability': group['Availability'].mean(),
                'Performance': group['Performance'].mean(), 'Quality': group['Quality'].mean(),
                'Produced_Units': prod, 'Scrapped_Units': scrap, 'Scrap_Rate': scrap_rate
            })
        df_mach_stats = pd.DataFrame(machine_stats)
        if not df_mach_stats.empty:
            top_machines = df_mach_stats.sort_values('OEE', ascending=False).head(5).to_dict(orient='records')
            bottom_machines = df_mach_stats.sort_values('OEE', ascending=True).head(5).to_dict(orient='records')

    # --- 5. STOP REASON DEEP DIVE ---
    stop_dur = df_s_curr.groupby('StopReason')['TotalStopDuration'].sum().sort_values(ascending=False)
    top_stops_dur = (stop_dur.head(10) / 3600).to_dict()
    stop_freq = df_s_curr['StopReason'].value_counts().head(10).to_dict()
    stop_dur_prev = df_s_prev.groupby('StopReason')['TotalStopDuration'].sum() / 3600
    emerging_issues = []
    for reason, curr_hours in top_stops_dur.items():
        prev_hours = stop_dur_prev.get(reason, 0)
        change = curr_hours - prev_hours
        if change > 0:
            emerging_issues.append({'Reason': reason, 'Current_Hours': curr_hours, 'Previous_Hours': prev_hours, 'Increase': change})
    
    # --- 5b. OPERATOR COMMENTS ANALYSIS ---
    comments_analysis = {}
    if 'Comment' in df_s_curr.columns:
        df_comments = df_s_curr[df_s_curr['Comment'].notna() & (df_s_curr['Comment'].str.strip() != '')].copy()
        if not df_comments.empty:
            # Top 5 Longest Comments
            df_comments['CommentLength'] = df_comments['Comment'].str.len()
            longest = df_comments.sort_values('CommentLength', ascending=False).head(5)
            comments_analysis['Top_5_Longest_Comments'] = longest[['Comment', 'StopReason', 'TotalStopDuration', 'IntervalStart']].to_dict(orient='records')
            # Format dates and duration
            for item in comments_analysis['Top_5_Longest_Comments']:
                item['IntervalStart'] = str(item['IntervalStart'])
                item['Duration_Minutes'] = round(item.pop('TotalStopDuration') / 60, 1)

            # Top 5 Recurring Comments
            recurring = df_comments['Comment'].str.strip().value_counts().head(5)
            comments_analysis['Top_5_Recurring_Comments'] = recurring.to_dict()

    # --- 6. SHIFT ANALYSIS ---
    shift_insights = {}
    if 'Shift' in df_full_oee.columns:
        max_date = df_full_oee['IntervalStart'].max()
        windows = {'Last_7_Days': max_date - pd.Timedelta(days=7), 'Last_30_Days': max_date - pd.Timedelta(days=30), 'Last_90_Days': max_date - pd.Timedelta(days=90)}
        unique_shifts = df_full_oee['Shift'].unique()
        if len(unique_shifts) <= 20:
            for shift in unique_shifts:
                if pd.isna(shift): continue
                shift_data = {}
                for label, start_dt in windows.items():
                    mask = (df_full_oee['Shift'] == shift) & (df_full_oee['IntervalStart'] >= start_dt)
                    subset = df_full_oee[mask]
                    if not subset.empty: shift_data[label] = subset['OEE'].mean()
                    else: shift_data[label] = None
                if shift_data.get('Last_30_Days') and shift_data.get('Last_90_Days'):
                    diff = shift_data['Last_30_Days'] - shift_data['Last_90_Days']
                    shift_data['Trend_vs_90d'] = f"{diff:+.1f}%"
                shift_insights[str(shift)] = shift_data

    # NEW: Current Period Shift Performance
    current_shift_performance = {}
    if df_oee_curr is not None and not df_oee_curr.empty and 'Shift' in df_oee_curr.columns:
            shift_stats = df_oee_curr.groupby('Shift').agg({
            'OEE': 'mean',
            'Availability': 'mean',
            'Performance': 'mean',
            'Quality': 'mean',
            'ProducedUnits': 'sum',
            'StopDuration': 'sum'
        })
            shift_stats['StopHours'] = shift_stats['StopDuration'] / 3600
            shift_stats = shift_stats.drop(columns=['StopDuration'])
            current_shift_performance = shift_stats.to_dict(orient='index')

    # --- 7. SCRAP RATE ANALYSIS ---
    scrap_analysis = {}
    if df_oee_curr is not None and not df_oee_curr.empty:
        if 'ArticleName' in df_oee_curr.columns:
             art_scrap = df_oee_curr.groupby('ArticleName')[['ProducedUnits', 'ScrappedUnits']].sum()
             art_scrap = art_scrap[art_scrap['ProducedUnits'] > 50]
             art_scrap['ScrapRate'] = (art_scrap['ScrappedUnits'] / art_scrap['ProducedUnits'] * 100).fillna(0)
             scrap_analysis['Top_5_High_Scrap_Articles'] = art_scrap.sort_values('ScrapRate', ascending=False).head(5).to_dict(orient='index')
        mach_scrap = df_oee_curr.groupby('MeasurePoint')[['ProducedUnits', 'ScrappedUnits']].sum()
        mach_scrap['ScrapRate'] = (mach_scrap['ScrappedUnits'] / mach_scrap['ProducedUnits'] * 100).fillna(0)
        scrap_analysis['Top_5_High_Scrap_Machines'] = mach_scrap.sort_values('ScrapRate', ascending=False).head(5).to_dict(orient='index')

    # --- 8. CONSTRUCT FINAL JSON ---
    data_summary = {
        "Meta": {
            "Company": company_name,
            "Analysis_Period": f"{current_start.date()} to {current_end.date()}",
            "Comparison_Period": f"{previous_start.date()} to {previous_end.date()}",
            "Data_Points_Analyzed": len(df_oee_curr),
            "Excluded_Stop_Reasons": exclude_reasons if exclude_reasons else [],
            "Excluded_Shifts": exclude_shifts if exclude_shifts else []
        },
        "Executive_Summary_KPIs": {
            "Current": current_metrics,
            "Change_vs_Previous": metrics_change
        },
        "Trends": {
            "OEE_Trend_Summary": trend_summary
        },
        "Machine_Performance": {
            "Top_5_Performers": top_machines,
            "Bottom_5_Performers": bottom_machines
        },
        "Scrap_Analysis": scrap_analysis,
        "Stop_Loss_Analysis": {
            "Top_10_By_Duration_Hours": top_stops_dur,
            "Top_10_By_Frequency_Count": stop_freq,
            "Emerging_Issues_Rising_Stops": emerging_issues
        },
        "Operator_Comments_Analysis": comments_analysis,
        "Shift_Performance_Current": current_shift_performance,
        "Shift_Performance_LongTerm": shift_insights
    }
    
    data_summary = recursive_round(data_summary)
    
    json_filename = f"{company_name}_ai_data.json"
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data_summary, f, indent=2, default=str)
        print(f"Data summary exported to '{json_filename}'")
    except Exception as e:
        print(f"Failed to export JSON data: {e}")
    
    # Load Report Template
    if include_recommendations:
        template_filename = "daily_production_report_template_with_recommendations.md"
    else:
        template_filename = "daily_production_report_template.md"

    template_path = os.path.join("templates", template_filename)
    report_template = ""
    try:
        with open(template_path, "r") as f:
            report_template = f.read()
    except Exception as e:
        print(f"Warning: Could not load template: {e}")
        report_template = "Template not found. Please generate a comprehensive report based on the data."

    # --- Generate Dynamic Shift Table Rows ---
    # We pre-fill the shift table rows to ensure dynamic handling of any number of shifts
    shift_rows = ""
    if current_shift_performance:
        # Sort shifts if possible (e.g., Shift 1, Shift 2)
        sorted_shifts = sorted(current_shift_performance.keys())
        for shift_name in sorted_shifts:
            stats = current_shift_performance[shift_name]
            # Get trend if available
            trend_val = "N/A"
            if shift_insights and str(shift_name) in shift_insights:
                 trend_val = shift_insights[str(shift_name)].get('Trend_vs_90d', 'N/A')
            
            # Format values
            oee = f"{stats.get('OEE', 0):.1f}%"
            avail = f"{stats.get('Availability', 0):.1f}%"
            perf = f"{stats.get('Performance', 0):.1f}%"
            qual = f"{stats.get('Quality', 0):.1f}%"
            prod = f"{int(stats.get('ProducedUnits', 0))}"
            stop_hrs = f"{stats.get('StopHours', 0):.1f}"
            
            row = f"| {shift_name} | {oee} | {avail} | {perf} | {qual} | {prod} | {stop_hrs} | {trend_val} |\n"
            shift_rows += row
    
    if not shift_rows:
        shift_rows = "| No Shift Data | - | - | - | - | - | - | - |"

    # Inject the dynamic table rows into the template
    report_template = report_template.replace("{{Shift_Performance_Table_Rows}}", shift_rows)

    prompt = f"""
    You are a senior production data analyst. Your task is to generate a daily production report based on the provided data summary, strictly following the provided Markdown template.
    
    DATA SUMMARY:
    {json.dumps(data_summary, indent=2, default=str)}
    
    REPORT TEMPLATE:
    {report_template}
    
    INSTRUCTIONS:
    1. **TEMPLATE STRUCTURE:** Follow the provided Markdown template structure exactly.
    2. **DATA FILLING:** Replace all data placeholders (e.g., {{OEE_Percentage}}, {{Top_Stop_Reason_1}}) with the corresponding values from the DATA SUMMARY.
    3. **INSIGHT GENERATION (CRITICAL):**
       - The template contains specific placeholders for analysis: {{Section_1_Insights}}, {{Section_2_Insights}}, etc.
       - These are NOT data variables; they are locations for you to write brief, high-value analysis.
       - **ACTION:** For each of these placeholders, write 1-3 sentences highlighting a key trend, anomaly, or cause relevant to that section.
       - **FALLBACK:** If there is nothing significant to note for a section, **YOU MUST REMOVE THE PLACEHOLDER**.
       - **PROHIBITION:** NEVER output the text "{{Section_X_Insights}}" in the final report. It must be replaced by text or removed.
    4. The 'Shift Performance Breakdown' table has been pre-filled with data rows. Do not alter the values in the table, but use them for your analysis.
    5. For the final "AI Strategic Insights & Recommendations" section, write a comprehensive deep-dive analysis, highlighting key trends, issues, and opportunities.
    6. For lists (e.g., Top Stop Reasons), fill them with the data provided.
    7. Tone: Professional, factual, concise, and objective.
    8. NO emojis.
    9. NO conversational filler.
    10. **LANGUAGE:** All generated text, analysis, and insights MUST be in English. If the input data (e.g., operator comments) is in another language (like Swedish), keep the original text for the data points but ensure your analysis and commentary are in English.
    """

    if include_recommendations:
        prompt += """
    11. **STRATEGIC RECOMMENDATIONS:**
        - The template contains a "Strategic Recommendations" section with placeholders {{Recommendation_1}}, {{Recommendation_2}}, {{Recommendation_3}}.
        - Provide 3 concrete, actionable, and data-driven recommendations based on your analysis.
        - Replace the placeholders with your recommendations.
        """
    
    print("Querying AI Model (this may take a few seconds)...")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "You are a helpful industrial data analyst assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        report_content = response.choices[0].message.content
        
        report_data = {
            "metadata": {
                "company": company_name,
                "analysis_period": f"{current_start.date()} to {current_end.date()}",
                "generated_at": pd.Timestamp.now().isoformat()
            },
            "report_content": report_content
        }
        
        report_filename_json = f"{company_name}_ai_report.json"
        report_filename_md = f"{company_name}_ai_report.md"
        try:
            with open(report_filename_json, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            print(f"Final report exported to '{report_filename_json}'")
            
            with open(report_filename_md, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Final report exported to '{report_filename_md}'")
        except Exception as e:
            print(f"Failed to export report: {e}")
            
        return report_content
    except Exception as e:
        return f"Error querying OpenAI API: {str(e)}"
