import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from src.analysis.oee import plot_oee_analysis, plot_shift_analysis
from src.analysis.stop_analysis import plot_stop_analysis, plot_operator_comment_analysis
from src.analysis.efficiency import plot_efficiency_trends, plot_machine_activity
from src.analysis.production import plot_production_output, plot_scrap_rate_analysis
from src.analysis.ai_insights import generate_ai_insights

def create_ui(data_by_company):
    # Extract data for easy access
    pr_data = data_by_company.get('pr', {})
    si_data = data_by_company.get('si', {})

    df_pr_setting = pr_data.get('setting')
    df_pr_worktime = pr_data.get('worktime')
    df_pr_stop = pr_data.get('stop')

    df_si_setting = si_data.get('setting')
    df_si_worktime = si_data.get('worktime')
    df_si_stop = si_data.get('stop')

    # Get all unique stop reasons
    all_stop_reasons_pr = sorted(df_pr_stop['StopReason'].unique().tolist()) if df_pr_stop is not None else []
    all_stop_reasons_si = sorted(df_si_stop['StopReason'].unique().tolist()) if df_si_stop is not None else []

    # Widgets
    company_selector = widgets.Dropdown(
        options=['pr', 'si'],
        value='pr',
        description='Company:',
        style={'description_width': 'initial'}
    )

    # Analysis Options split into 3 rows
    options_row1 = ['OEE Overview', 'Stop Analysis', 'Efficiency Analysis', 'Machine Activity']
    options_row2 = ['Production Output', 'Shift Analysis', 'Scrap Rate Analysis', 'Operator Comments']
    options_row3 = ['AI Insights']

    # Use percentage widths for responsiveness
    style_row_12 = {'button_width': '200px'} # Fixed pixel width per button
    style_row_3 = {'button_width': '812px'} # Fixed pixel width for AI button

    # Layout with minimum width to prevent buttons from becoming too small
    row_layout = widgets.Layout(width='100%', display='flex', justify_content='center')

    analysis_row1 = widgets.ToggleButtons(
        options=options_row1,
        value=None,
        button_style='info',
        style=style_row_12,
        layout=row_layout
    )
    
    analysis_row2 = widgets.ToggleButtons(
        options=options_row2,
        value=None,
        button_style='info',
        style=style_row_12,
        layout=row_layout
    )

    analysis_row3 = widgets.ToggleButtons(
        options=options_row3,
        value='AI Insights',
        button_style='success', # Distinct style for AI
        style=style_row_3,
        icons=['magic'],
        layout=row_layout
    )

    # Logic to ensure only one is selected across all rows
    def on_row1_change(change):
        if change['new']:
            analysis_row2.value = None
            analysis_row3.value = None
            update_selectors()

    def on_row2_change(change):
        if change['new']:
            analysis_row1.value = None
            analysis_row3.value = None
            update_selectors()

    def on_row3_change(change):
        if change['new']:
            analysis_row1.value = None
            analysis_row2.value = None
            update_selectors()

    analysis_row1.observe(on_row1_change, names='value')
    analysis_row2.observe(on_row2_change, names='value')
    analysis_row3.observe(on_row3_change, names='value')

    # Helper to get current value
    def get_analysis_value():
        if analysis_row1.value: return analysis_row1.value
        if analysis_row2.value: return analysis_row2.value
        if analysis_row3.value: return analysis_row3.value
        return 'OEE Overview' # Default fallback

    start_date_picker = widgets.DatePicker(
        description='Start Date',
        value=pd.to_datetime('2025-08-01').date(),
        disabled=False
    )

    end_date_picker = widgets.DatePicker(
        description='End Date',
        value=pd.to_datetime('2025-09-01').date(),
        disabled=False
    )

    # Stop Reason Selector (Multi-select)
    stop_reason_selector = widgets.SelectMultiple(
        options=all_stop_reasons_pg,
        value=[],
        description='',
        disabled=True, # Only enabled for Stop Analysis
        layout={'height': '150px', 'width': '400px'}
    )

    # Granularity Selector
    granularity_selector = widgets.Dropdown(
        options=['Day', 'Week', 'Month', 'Year'],
        value='Week',
        description='',
        disabled=True # Only for Stop Analysis
    )

    # Exclude Shift Selector
    exclude_shift_selector = widgets.SelectMultiple(
        options=[],
        value=[],
        description='',
        disabled=False,
        layout={'height': '150px', 'width': '400px'}
    )

    # Include Recommendations Checkbox
    include_recommendations_checkbox = widgets.Checkbox(
        value=True,
        description='Include Recommendations',
        disabled=True, # Only enabled for AI Insights
        indent=False
    )

    execute_button = widgets.Button(
        description=' Run Analysis',
        button_style='primary',
        icon='play',
        layout={'width': '200px'}
    )

    output_area = widgets.Output()

    def update_selectors(*args):
        company = company_selector.value
        analysis = get_analysis_value()
        
        # Update Stop Reasons
        if company == 'pr':
            stop_reason_selector.options = all_stop_reasons_pr
            # Update Shifts
            if df_pr_worktime is not None and 'Shift' in df_pr_worktime.columns:
                shifts = sorted(df_pr_worktime['Shift'].dropna().unique().tolist())
                exclude_shift_selector.options = shifts
        else:
            stop_reason_selector.options = all_stop_reasons_si
            # Update Shifts
            if df_si_worktime is not None and 'Shift' in df_si_worktime.columns:
                shifts = sorted(df_si_worktime['Shift'].dropna().unique().tolist())
                exclude_shift_selector.options = shifts
                
        # Enable/Disable widgets based on analysis type
        if analysis == 'Stop Analysis' or analysis == 'AI Insights':
            stop_reason_selector.disabled = False
            granularity_selector.disabled = False if analysis == 'Stop Analysis' else True
        elif analysis == 'Scrap Rate Analysis' or analysis == 'Efficiency Analysis':
            stop_reason_selector.disabled = True
            granularity_selector.disabled = False
        else:
            stop_reason_selector.disabled = True
            granularity_selector.disabled = True
        
        # Enable/Disable Recommendations Checkbox
        if analysis == 'AI Insights':
            include_recommendations_checkbox.disabled = False
        else:
            include_recommendations_checkbox.disabled = True

    company_selector.observe(update_selectors, 'value')
    # analysis_type.observe(update_selectors, 'value') # Handled by individual row observers

    def on_execute_button_clicked(b):
        output_area.clear_output()
        
        company = company_selector.value
        analysis = get_analysis_value()
        start_date = start_date_picker.value
        end_date = end_date_picker.value
        exclude_reasons = list(stop_reason_selector.value) if stop_reason_selector.value else None
        granularity = granularity_selector.value
        exclude_shifts = list(exclude_shift_selector.value) if exclude_shift_selector.value else None
        include_recommendations = include_recommendations_checkbox.value
        
        # Select Data
        if company == 'pr':
            df_w = df_pr_worktime
            df_s = df_pr_stop
            df_set = df_pr_setting
        else:
            df_w = df_si_worktime
            df_s = df_si_stop
            df_set = df_si_setting
            
        with output_area:
            if analysis == 'OEE Overview':
                plot_oee_analysis(df_w, df_set, company, start_date, end_date, exclude_shifts)
            elif analysis == 'Stop Analysis':
                plot_stop_analysis(df_s, company, start_date, end_date, exclude_reasons, granularity, exclude_shifts)
            elif analysis == 'Efficiency Analysis':
                plot_efficiency_trends(df_w, company, start_date, end_date, exclude_shifts, granularity)
            elif analysis == 'Machine Activity':
                plot_machine_activity(df_w, company, start_date, end_date, exclude_shifts)
            elif analysis == 'Production Output':
                plot_production_output(df_w, company, start_date, end_date, exclude_shifts)
            elif analysis == 'Shift Analysis':
                plot_shift_analysis(df_w, df_set, company, start_date, end_date, exclude_shifts)
            elif analysis == 'Scrap Rate Analysis':
                plot_scrap_rate_analysis(df_w, company, start_date, end_date, exclude_shifts, granularity)
            elif analysis == 'Operator Comments':
                plot_operator_comment_analysis(df_s, company, start_date, end_date, exclude_shifts)
            elif analysis == 'AI Insights':
                generate_ai_insights(df_w, df_s, df_set, company, start_date, end_date, exclude_reasons, exclude_shifts, include_recommendations)

    execute_button.on_click(on_execute_button_clicked)

    # Initial update
    update_selectors()

    # Layout
    header = widgets.HTML("<h2>GoodSolutions Analytics Dashboard</h2>")
    
    config_box = widgets.HBox([
        company_selector,
        start_date_picker,
        end_date_picker
    ], layout=widgets.Layout(align_items='center', margin='10px 0 20px 0'))
    
    analysis_box = widgets.VBox([
        widgets.HTML("<b>Select Analysis Module:</b>"),
        analysis_row1,
        analysis_row2,
        analysis_row3
    ], layout=widgets.Layout(margin='0 0 20px 0'))
    
    filters_content = widgets.VBox([
        widgets.HBox([
            widgets.VBox([widgets.Label("Exclude Stop Reasons:"), stop_reason_selector]),
            widgets.VBox([widgets.Label("Exclude Shifts:"), exclude_shift_selector], layout=widgets.Layout(margin='0 0 0 40px'))
        ]),
        widgets.HBox([
            widgets.VBox([widgets.Label("Granularity:"), granularity_selector], layout=widgets.Layout(margin='20px 0 0 0')),
            widgets.VBox([widgets.Label("AI Options:"), include_recommendations_checkbox], layout=widgets.Layout(margin='20px 0 0 40px'))
        ])
    ])
    
    filters_accordion = widgets.Accordion(children=[filters_content])
    filters_accordion.set_title(0, 'Advanced Filters (Stop Reasons, Shifts, Granularity)')
    filters_accordion.selected_index = None 
    
    action_box = widgets.HBox([execute_button], layout=widgets.Layout(justify_content='center', margin='20px 0'))

    ui = widgets.VBox([
        header,
        config_box,
        analysis_box,
        filters_accordion,
        action_box,
        widgets.HTML("<hr>"),
        output_area
    ], layout=widgets.Layout(padding='20px', border='1px solid #e0e0e0', width='95%'))

    display(ui)
