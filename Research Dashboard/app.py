import seaborn as sns
import pandas as pd
from shared import df, filename
import plotly.express as px
from shinywidgets import render_widget  
from Regmodels import OLS
from shiny import reactive
from shiny.express import input, render, ui
import io
import base64
import logging
import traceback
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration and styling
CUSTOM_COLORS = {
    'primary': '#5040ff',
    'secondary': '#e6bc32', 
    'accent': '#ab265f',
    'background': '#f9f9f9',
    'grid': 'lightgray',
    'text': '#2c3e50'
}

PLOTLY_TEMPLATE = {
    'template': 'plotly_white',
    'plot_bgcolor': CUSTOM_COLORS['background'],
    'font': {'family': 'Arial', 'size': 14, 'color': CUSTOM_COLORS['text']},
    'title_font': {'size': 20, 'family': 'Verdana', 'color': CUSTOM_COLORS['text']},
    'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40},
    'xaxis': {'showgrid': True, 'gridcolor': CUSTOM_COLORS['grid'], 'zeroline': False},
    'yaxis': {'showgrid': True, 'gridcolor': CUSTOM_COLORS['grid'], 'zeroline': False}
}

# Reactive values
model_store = reactive.Value()
current_df = reactive.Value(df)
current_filename = reactive.Value(filename)
selected_columns_store = reactive.Value([])  # Store selected columns as a list
loaded_state_store = reactive.Value(None)  # Store loaded state for UI updates
input_state_store = reactive.Value({})  # Store all input values as they change
save_files_trigger = reactive.Value(0)  # Trigger to refresh save files list
filtered_df = reactive.Value(None)  # Store filtered dataframe (complete cases only)

ui.page_opts(title="Analytics Dashboard")

# Error handling utility functions
def safe_get_dataframe():
    """Safely get the current dataframe with error handling"""
    try:
        # Check if complete cases filter is enabled
        if hasattr(input, 'show_complete_cases') and input.show_complete_cases():
            # Use filtered dataframe if available
            filtered = filtered_df.get()
            if filtered is not None and not filtered.empty:
                return filtered
            else:
                # Create filtered dataframe if not already created
                df = current_df.get()
                if df is None or df.empty:
                    return pd.DataFrame()
                
                # Get selected columns to filter on
                selected_cols = selected_columns_store.get()
                if not selected_cols:
                    # If no columns selected, filter on all columns
                    filtered_data = df.dropna()
                else:
                    # Filter on selected columns only
                    valid_cols = [col for col in selected_cols if col in df.columns]
                    if not valid_cols:
                        return pd.DataFrame()
                    filtered_data = df.dropna(subset=valid_cols)
                
                # Store the filtered dataframe for future use
                filtered_df.set(filtered_data)
                return filtered_data
        else:
            # Return original dataframe
            df = current_df.get()
            if df is None or df.empty:
                return pd.DataFrame()
            return df
    except Exception as e:
        logger.error(f"Error getting dataframe: {e}")
        return pd.DataFrame()

def safe_get_columns():
    """Safely get columns from the dataframe"""
    try:
        df = safe_get_dataframe()
        if df.empty:
            return []
        return list(df.columns)
    except Exception as e:
        logger.error(f"Error getting columns: {e}")
        return []

def validate_numeric_input(value, default_value, min_value=None, max_value=None):
    """Validate and convert numeric input with error handling"""
    try:
        if value is None or value == "":
            return default_value
        
        num_value = float(value)
        
        if min_value is not None and num_value < min_value:
            ui.notification_show(f"Value must be at least {min_value}. Using default value {default_value}.", type="warning")
            return default_value
        
        if max_value is not None and num_value > max_value:
            ui.notification_show(f"Value must be at most {max_value}. Using default value {default_value}.", type="warning")
            return default_value
        
        return num_value
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid numeric input '{value}': {e}. Using default value {default_value}.")
        ui.notification_show(f"Invalid input '{value}'. Using default value {default_value}.", type="warning")
        return default_value

def safe_column_access(df, columns):
    """Safely access columns from dataframe"""
    try:
        if isinstance(columns, str):
            columns = [columns]
        
        available_cols = [col for col in columns if col in df.columns]
        if not available_cols:
            logger.warning(f"No valid columns found among {columns}")
            return pd.DataFrame()
        
        return df[available_cols]
    except Exception as e:
        logger.error(f"Error accessing columns {columns}: {e}")
        return pd.DataFrame()

# Utility functions
def get_columns():
    """Get all columns from the dataframe"""
    return safe_get_columns()

def get_selected_columns():
    """Get selected columns with fallback to all columns"""
    try:
        selected = selected_columns_store.get()
        all_cols = get_columns()
        if not selected or not all_cols:
            return all_cols
        return [col for col in selected if col in all_cols]
    except Exception as e:
        logger.error(f"Error getting selected columns: {e}")
        return get_columns()

def get_numeric_columns(columns=None):
    """Get numeric columns from selected or all columns"""
    try:
        if columns is None:
            columns = get_selected_columns()
        
        df = safe_get_dataframe()
        if df.empty:
            return []
        
        numeric_cols = df[columns].select_dtypes(include=['float64', 'int64']).columns.tolist()
        return numeric_cols
    except Exception as e:
        logger.error(f"Error getting numeric columns: {e}")
        return []

def get_year_columns():
    """Get columns that are likely to be date/year columns"""
    try:
        cols = []
        df = safe_get_dataframe()
        if df.empty:
            return cols
            
        for col in get_selected_columns():
            try:
                if (col in df.columns and 
                    (df[col].dtype in ["int64", "float64"] or
                     isinstance(col, str) and any(keyword in col.lower() for keyword in ["date", "year", "time"]))):
                    cols.append(col)
            except Exception as e:
                logger.warning(f"Error checking column {col}: {e}")
                continue
        return cols
    except Exception as e:
        logger.error(f"Error getting year columns: {e}")
        return []

def apply_plotly_styling(fig, title=None, x_title=None, y_title=None):
    """Apply consistent styling to plotly figures"""
    try:
        if fig is None:
            return None
            
        fig.update_layout(**PLOTLY_TEMPLATE)
        if title:
            fig.update_layout(title=title)
        if x_title:
            fig.update_layout(xaxis_title=x_title)
        if y_title:
            fig.update_layout(yaxis_title=y_title)
        return fig
    except Exception as e:
        logger.error(f"Error applying plotly styling: {e}")
        return fig

def process_panel_data(year_col, y_col, aggregation_option, filter_var="None", filter_value=None):
    """Process panel data with filtering and aggregation"""
    try:
        df = safe_get_dataframe()
        if df.empty:
            return pd.DataFrame(), 0, 0, None, None
        
        # Validate required columns exist
        required_cols = [year_col, y_col]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(), 0, 0, None, None
        
        cols = [year_col, y_col]
        if filter_var not in (None, "None", "") and filter_var not in cols and filter_var in df.columns:
            cols.append(filter_var)
        
        dff = df[cols].copy()
        dropped_na = dff.isna().any(axis=1).sum()
        dff = dff.dropna()
        
        if dff.empty:
            logger.warning("No data remaining after removing NA values")
            return dff, 0, dropped_na, year_col, y_col
        
        # Apply filter
        dropped_filter = 0
        if filter_var not in (None, "None", "") and filter_value not in (None, ""):
            try:
                before_filter = len(dff)
                dff = dff[dff[filter_var].astype(str) == str(filter_value)]
                dropped_filter = before_filter - len(dff)
            except Exception as e:
                logger.error(f"Error applying filter: {e}")
                dropped_filter = 0
        
        # Apply aggregation
        dropped_agg = 0
        group_col = year_col
        
        if aggregation_option in ["by_year_sum", "by_year_mean"]:
            try:
                dff["_year"] = pd.to_datetime(dff[year_col], errors="coerce").dt.year
                dropped_agg = dff["_year"].isna().sum()
                dff = dff.dropna(subset=["_year"])
                group_col = "_year"
                
                agg_func = 'sum' if aggregation_option == "by_year_sum" else 'mean'
                dff = dff.groupby(group_col, as_index=False)[y_col].agg(agg_func)
            except Exception as e:
                logger.error(f"Error in year aggregation: {e}")
                return pd.DataFrame(), 0, dropped_na + dropped_filter, year_col, y_col
        elif aggregation_option == "by_date":
            try:
                dff = dff.groupby(group_col, as_index=False)[y_col].sum()
            except Exception as e:
                logger.error(f"Error in date aggregation: {e}")
                return pd.DataFrame(), 0, dropped_na + dropped_filter, year_col, y_col
        
        if dff.empty:
            return dff, 0, dropped_na + dropped_filter + dropped_agg, group_col, y_col
        
        dff = dff.sort_values(by=group_col)
        dff = dff[[group_col, y_col]]
        
        n_obs = len(dff)
        n_dropped = dropped_na + dropped_filter + dropped_agg
        
        return dff, n_obs, n_dropped, group_col, y_col
        
    except Exception as e:
        logger.error(f"Error in process_panel_data: {e}")
        return pd.DataFrame(), 0, 0, None, None

def process_ratio_data(year_col, numerator_col, denominator_col, aggregation_option, filter_var="None", filter_value=None):
    """Process ratio data with filtering and aggregation"""
    try:
        df = safe_get_dataframe()
        if df.empty:
            return pd.DataFrame(), 0, 0, None, None
        
        # Validate required columns exist
        required_cols = [year_col, numerator_col, denominator_col]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame(), 0, 0, None, None
        
        cols = [year_col, numerator_col, denominator_col]
        if filter_var not in (None, "None", "") and filter_var not in cols and filter_var in df.columns:
            cols.append(filter_var)
        
        dff = df[cols].copy()
        dropped_na = dff.isna().any(axis=1).sum()
        dff = dff.dropna()
        
        if dff.empty:
            logger.warning("No data remaining after removing NA values")
            return dff, 0, dropped_na, year_col, 'ratio'
        
        # Check for division by zero
        if (dff[denominator_col] == 0).any():
            logger.warning("Division by zero detected in denominator column")
            dff = dff[dff[denominator_col] != 0]
            dropped_na += len(dff[dff[denominator_col] == 0])
        
        # Apply filter
        dropped_filter = 0
        if filter_var not in (None, "None", "") and filter_value not in (None, ""):
            try:
                before_filter = len(dff)
                dff = dff[dff[filter_var].astype(str) == str(filter_value)]
                dropped_filter = before_filter - len(dff)
            except Exception as e:
                logger.error(f"Error applying filter: {e}")
                dropped_filter = 0
        
        # Apply aggregation
        dropped_agg = 0
        group_col = year_col
        
        if aggregation_option in ["by_year_sum", "by_year_mean"]:
            try:
                dff["_year"] = pd.to_datetime(dff[year_col], errors="coerce").dt.year
                dropped_agg = dff["_year"].isna().sum()
                dff = dff.dropna(subset=["_year"])
                group_col = "_year"
                
                agg_func = 'sum' if aggregation_option == "by_year_sum" else 'mean'
                dff = dff.groupby(group_col, as_index=False)[[numerator_col, denominator_col]].agg(agg_func)
            except Exception as e:
                logger.error(f"Error in year aggregation: {e}")
                return pd.DataFrame(), 0, dropped_na + dropped_filter, year_col, 'ratio'
        elif aggregation_option == "by_date":
            try:
                dff = dff.groupby(group_col, as_index=False)[[numerator_col, denominator_col]].sum()
            except Exception as e:
                logger.error(f"Error in date aggregation: {e}")
                return pd.DataFrame(), 0, dropped_na + dropped_filter, year_col, 'ratio'
        
        if dff.empty:
            return dff, 0, dropped_na + dropped_filter + dropped_agg, group_col, 'ratio'
        
        dff = dff.sort_values(by=group_col)
        
        # Calculate ratio with error handling
        try:
            dff['ratio'] = dff[numerator_col] / dff[denominator_col]
            # Remove infinite values
            dff = dff[~dff['ratio'].isin([float('inf'), float('-inf')])]
        except Exception as e:
            logger.error(f"Error calculating ratio: {e}")
            return pd.DataFrame(), 0, dropped_na + dropped_filter + dropped_agg, group_col, 'ratio'
        
        dff = dff[[group_col, 'ratio']]
        
        n_obs = len(dff)
        n_dropped = dropped_na + dropped_filter + dropped_agg
        
        return dff, n_obs, n_dropped, group_col, 'ratio'
        
    except Exception as e:
        logger.error(f"Error in process_ratio_data: {e}")
        return pd.DataFrame(), 0, 0, None, None

# State management functions
def get_current_state():
    """Capture the current state of all inputs"""
    try:
        state = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'filename': current_filename.get(),
                'rows': len(current_df.get()) if current_df.get() is not None else 0,
                'columns': list(current_df.get().columns) if current_df.get() is not None else []
            },
            'selected_columns': selected_columns_store.get(),
            'inputs': input_state_store.get() or {}
        }
        
        return state
    except Exception as e:
        logger.error(f"Error capturing current state: {e}")
        return None

def save_state_to_file(state, filename):
    """Save state to a JSON file"""
    try:
        states_dir = "states"
        if not os.path.exists(states_dir):
            os.makedirs(states_dir)
        
        filepath = os.path.join(states_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving state to file: {e}")
        return False

def load_state_from_file(filename):
    """Load state from a JSON file"""
    try:
        filepath = os.path.join("states", filename)
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        return state
    except Exception as e:
        logger.error(f"Error loading state from file: {e}")
        return None

def apply_state_to_inputs(state):
    """Apply loaded state to current inputs"""
    try:
        if not state or 'inputs' not in state:
            return False
        
        # Apply selected columns
        if 'selected_columns' in state:
            selected_columns_store.set(state['selected_columns'])
        
        # Store the loaded state for UI reference
        loaded_state_store.set(state)
        
        # Note: In Shiny, we can't directly set input values from reactive effects
        # The inputs will be restored when the user loads a save file
        # The loaded state is stored in loaded_state_store for UI components to reference
        return True
    except Exception as e:
        logger.error(f"Error applying state to inputs: {e}")
        return False

def get_save_files():
    """Get list of available save files"""
    try:
        # Trigger reactive update
        save_files_trigger.get()
        
        states_dir = "states"
        if not os.path.exists(states_dir):
            return []
        
        save_files = []
        for filename in os.listdir(states_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(states_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        state = json.load(f)
                    save_files.append({
                        'filename': filename,
                        'timestamp': state.get('timestamp', 'Unknown'),
                        'dataset': state.get('dataset_info', {}).get('filename', 'Unknown'),
                        'rows': state.get('dataset_info', {}).get('rows', 0)
                    })
                except Exception as e:
                    logger.warning(f"Could not read save file {filename}: {e}")
                    continue
        
        # Sort by timestamp (newest first)
        save_files.sort(key=lambda x: x['timestamp'], reverse=True)
        return save_files
    except Exception as e:
        logger.error(f"Error getting save files: {e}")
        return []

def trigger_save_files_refresh():
    """Trigger a refresh of the save files list"""
    try:
        current_trigger = save_files_trigger.get()
        save_files_trigger.set(current_trigger + 1)
    except Exception as e:
        logger.error(f"Error triggering save files refresh: {e}")

def get_loaded_state_value(input_name, default_value=None):
    """Get a value from the loaded state for UI display purposes"""
    try:
        loaded_state = loaded_state_store.get()
        if loaded_state and 'inputs' in loaded_state:
            return loaded_state['inputs'].get(input_name, default_value)
        return default_value
    except Exception as e:
        logger.error(f"Error getting loaded state value for {input_name}: {e}")
        return default_value

# Main UI Layout
ui.nav_spacer()

# Upload Data Panel
with ui.nav_panel("Upload Data"):
    with ui.card():
        ui.h3("Upload New Dataset")
        ui.markdown("Upload a CSV file to replace the current dataset.")
        
        ui.input_file(
            "file_upload",
            "Choose CSV file:",
            accept=[".csv"],
            multiple=False
        )
        
        @render.ui
        def upload_status():
            try:
                if input.file_upload() is not None:
                    return ui.markdown("✅ File uploaded successfully! Click 'Load Dataset' to apply changes.")
                return ui.markdown("📁 No file selected")
            except Exception as e:
                logger.error(f"Error in upload_status: {e}")
                return ui.markdown("❌ Error checking upload status")
        
        ui.input_action_button("load_dataset", "Load Dataset", class_="btn-primary")
        
        @render.ui
        def current_dataset_info():
            try:
                df = safe_get_dataframe()
                filename = current_filename.get() or "Unknown"
                rows = len(df) if not df.empty else 0
                cols = len(df.columns) if not df.empty else 0
                return ui.markdown(f"**Current dataset:** {filename} ({rows} rows, {cols} columns)")
            except Exception as e:
                logger.error(f"Error in current_dataset_info: {e}")
                return ui.markdown("**Current dataset:** Error loading dataset information")
        
        @reactive.Effect
        @reactive.event(input.load_dataset)
        def load_new_dataset():
            if input.file_upload() is not None:
                try:
                    # Read the uploaded file
                    file_info = input.file_upload()[0]
                    file_content = file_info['datapath']
                    
                    # Load the new dataset
                    new_df = pd.read_csv(file_content)
                    
                    # Validate the dataset
                    if new_df.empty:
                        ui.notification_show("Error: The uploaded file contains no data.", type="error")
                        return
                    
                    if len(new_df.columns) == 0:
                        ui.notification_show("Error: The uploaded file contains no columns.", type="error")
                        return
                    
                    new_filename = file_info['name']
                    
                    # Update reactive values
                    current_df.set(new_df)
                    current_filename.set(new_filename)
                    
                    # Reset model store and selected columns
                    model_store.set(None)
                    selected_columns_store.set([])  # Reset to show all columns
                    
                    ui.notification_show(f"Successfully loaded {new_filename} with {len(new_df)} rows and {len(new_df.columns)} columns.", type="success")
                    
                except pd.errors.EmptyDataError:
                    ui.notification_show("Error: The uploaded file is empty or contains no valid data.", type="error")
                except pd.errors.ParserError as e:
                    ui.notification_show(f"Error parsing CSV file: {str(e)}", type="error")
                except UnicodeDecodeError:
                    ui.notification_show("Error: The file encoding is not supported. Please save the file as UTF-8.", type="error")
                except Exception as e:
                    logger.error(f"Error loading file: {e}")
                    ui.notification_show(f"Error loading file: {str(e)}", type="error")

# Save/Load Panel
with ui.nav_panel("Save/Load"):
    with ui.navset_card_underline(title="Save/Load Dashboard State"):
        with ui.nav_panel("Save State"):
            with ui.card():
                ui.h3("💾 Save Current State")
                ui.markdown("Save your current dashboard configuration and selections.")
                
                ui.input_text(
                    "save_filename",
                    "Save filename:",
                    placeholder="my_dashboard_state",
                    value=f"dashboard_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                @render.ui
                def save_preview():
                    try:
                        state = get_current_state()
                        if state is None:
                            return ui.markdown("❌ Error generating save preview")
                        
                        dataset_info = state.get('dataset_info', {})
                        selected_cols = state.get('selected_columns', [])
                        inputs_count = len(state.get('inputs', {}))
                        
                        return ui.markdown(f"""
                        **Save Preview:**
                        - **Dataset:** {dataset_info.get('filename', 'Unknown')} ({dataset_info.get('rows', 0)} rows)
                        - **Selected Columns:** {len(selected_cols) if selected_cols else 'All'} columns
                        - **Input Settings:** {inputs_count} saved inputs
                        - **Timestamp:** {state.get('timestamp', 'Unknown')}
                        """)
                    except Exception as e:
                        logger.error(f"Error in save_preview: {e}")
                        return ui.markdown("❌ Error generating save preview")
                
                ui.input_action_button("save_state", "💾 Save State", class_="btn-success")
                
                @reactive.Effect
                @reactive.event(input.save_state)
                def save_current_state():
                    try:
                        filename = input.save_filename()
                        if not filename.strip():
                            ui.notification_show("Please enter a filename", type="warning")
                            return
                        
                        # Ensure filename has .json extension
                        if not filename.endswith('.json'):
                            filename += '.json'
                        
                        state = get_current_state()
                        if state is None:
                            ui.notification_show("Error: Could not capture current state", type="error")
                            return
                        
                        if save_state_to_file(state, filename):
                            ui.notification_show(f"✅ State saved successfully as {filename}", type="success")
                            # Trigger refresh of save files list
                            trigger_save_files_refresh()
                        else:
                            ui.notification_show("❌ Error saving state", type="error")
                            
                    except Exception as e:
                        logger.error(f"Error saving state: {e}")
                        ui.notification_show(f"Error saving state: {str(e)}", type="error")
        
        with ui.nav_panel("Load State"):
            with ui.card():
                ui.h3("📂 Load Saved State")
                ui.markdown("Load a previously saved dashboard configuration.")
                
                @render.ui
                def load_save_files():
                    try:
                        save_files = get_save_files()
                        if not save_files:
                            return ui.markdown("📁 No save files found")
                        
                        # Create file selection dropdown
                        choices = {}
                        for save_file in save_files:
                            timestamp = save_file['timestamp']
                            if timestamp != 'Unknown':
                                try:
                                    dt = datetime.fromisoformat(timestamp)
                                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                                except:
                                    formatted_time = timestamp
                            else:
                                formatted_time = 'Unknown'
                            
                            label = f"{save_file['filename']} - {save_file['dataset']} ({save_file['rows']} rows) - {formatted_time}"
                            choices[save_file['filename']] = label
                        
                        return ui.input_select(
                            "load_filename",
                            "Select save file:",
                            choices=choices,
                            selected=None
                        )
                    except Exception as e:
                        logger.error(f"Error loading save files: {e}")
                        return ui.markdown("❌ Error loading save files")
                
                @render.ui
                def load_preview():
                    try:
                        selected_file = input.load_filename()
                        if not selected_file:
                            return ui.markdown("📁 Select a save file to preview")
                        
                        state = load_state_from_file(selected_file)
                        if state is None:
                            return ui.markdown("❌ Error loading save file")
                        
                        dataset_info = state.get('dataset_info', {})
                        selected_cols = state.get('selected_columns', [])
                        inputs_count = len(state.get('inputs', {}))
                        
                        return ui.markdown(f"""
                        **Load Preview:**
                        - **Dataset:** {dataset_info.get('filename', 'Unknown')} ({dataset_info.get('rows', 0)} rows)
                        - **Selected Columns:** {len(selected_cols) if selected_cols else 'All'} columns
                        - **Input Settings:** {inputs_count} saved inputs
                        - **Timestamp:** {state.get('timestamp', 'Unknown')}
                        """)
                    except Exception as e:
                        logger.error(f"Error in load_preview: {e}")
                        return ui.markdown("❌ Error loading save preview")
                
                ui.input_action_button("load_state", "📂 Load State", class_="btn-primary")
                
                @render.ui
                def load_status():
                    try:
                        loaded_state = loaded_state_store.get()
                        if loaded_state is None:
                            return ui.markdown("📁 No state loaded")
                        
                        dataset_info = loaded_state.get('dataset_info', {})
                        selected_cols = loaded_state.get('selected_columns', [])
                        inputs_count = len(loaded_state.get('inputs', {}))
                        
                        return ui.markdown(f"""
                        **✅ State Loaded Successfully:**
                        - **Dataset:** {dataset_info.get('filename', 'Unknown')} ({dataset_info.get('rows', 0)} rows)
                        - **Selected Columns:** {len(selected_cols) if selected_cols else 'All'} columns restored
                        - **Input Settings:** {inputs_count} settings available
                        - **Loaded at:** {loaded_state.get('timestamp', 'Unknown')}
                        
                        *Note: Column selections have been restored. Input values are available for reference.*
                        """)
                    except Exception as e:
                        logger.error(f"Error in load_status: {e}")
                        return ui.markdown("❌ Error displaying load status")
                
                @reactive.Effect
                @reactive.event(input.load_state)
                def load_saved_state():
                    try:
                        selected_file = input.load_filename()
                        if not selected_file:
                            ui.notification_show("Please select a save file", type="warning")
                            return
                        
                        state = load_state_from_file(selected_file)
                        if state is None:
                            ui.notification_show("Error: Could not load save file", type="error")
                            return
                        
                        # Apply the loaded state
                        if apply_state_to_inputs(state):
                            ui.notification_show(f"✅ State loaded successfully from {selected_file}", type="success")
                        else:
                            ui.notification_show("⚠️ State loaded but some settings could not be applied", type="warning")
                            
                    except Exception as e:
                        logger.error(f"Error loading state: {e}")
                        ui.notification_show(f"Error loading state: {str(e)}", type="error")
        
        with ui.nav_panel("Manage Saves"):
            with ui.card():
                ui.h3("🗂️ Manage Save Files")
                ui.markdown("View and manage your saved dashboard states.")
                
                @render.data_frame
                def save_files_table():
                    try:
                        save_files = get_save_files()
                        if not save_files:
                            return pd.DataFrame({'Message': ['No save files found']})
                        
                        # Create a table of save files
                        table_data = []
                        for save_file in save_files:
                            timestamp = save_file['timestamp']
                            if timestamp != 'Unknown':
                                try:
                                    dt = datetime.fromisoformat(timestamp)
                                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                                except:
                                    formatted_time = timestamp
                            else:
                                formatted_time = 'Unknown'
                            
                            table_data.append({
                                'Filename': save_file['filename'],
                                'Dataset': save_file['dataset'],
                                'Rows': save_file['rows'],
                                'Created': formatted_time
                            })
                        
                        return pd.DataFrame(table_data)
                    except Exception as e:
                        logger.error(f"Error creating save files table: {e}")
                        return pd.DataFrame({'Error': [f'Error loading save files: {str(e)}']})
                
                # Delete functionality
                with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                    ui.h5("🗑️ Delete Save File")
                    ui.markdown("Remove a saved state file.")
                    
                    @render.ui
                    def delete_save_files():
                        try:
                            save_files = get_save_files()
                            if not save_files:
                                return ui.markdown("📁 No save files to delete")
                            
                            choices = {save_file['filename']: save_file['filename'] for save_file in save_files}
                            return ui.input_select(
                                "delete_filename",
                                "Select file to delete:",
                                choices=choices,
                                selected=None
                            )
                        except Exception as e:
                            logger.error(f"Error loading delete files: {e}")
                            return ui.markdown("❌ Error loading delete files")
                    
                    ui.input_action_button("delete_state", "🗑️ Delete File", class_="btn-danger")
                    
                    @reactive.Effect
                    @reactive.event(input.delete_state)
                    def delete_saved_state():
                        try:
                            selected_file = input.delete_filename()
                            if not selected_file:
                                ui.notification_show("Please select a file to delete", type="warning")
                                return
                            
                            filepath = os.path.join("states", selected_file)
                            if not os.path.exists(filepath):
                                ui.notification_show("File not found", type="error")
                                return
                            
                            os.remove(filepath)
                            ui.notification_show(f"✅ File {selected_file} deleted successfully", type="success")
                            # Trigger refresh of save files list
                            trigger_save_files_refresh()
                            
                        except Exception as e:
                            logger.error(f"Error deleting state: {e}")
                            ui.notification_show(f"Error deleting file: {str(e)}", type="error")

# Raw Data Panel
with ui.nav_panel("Raw Data"):
    with ui.navset_card_underline(title="Raw Data"):
        with ui.nav_panel("Table"):
            with ui.card():
                ui.h4("📊 Column Selection")
                ui.markdown("Choose which columns to display and use for your analysis.")
                
                # Column statistics and quick actions
                with ui.layout_column_wrap(width=1/3, gap="1rem"):
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("📋 Dataset Info")
                        @render.ui
                        def dataset_stats():
                            try:
                                df = safe_get_dataframe()
                                columns = get_columns()
                                if df.empty:
                                    return ui.markdown("**Total:** 0 columns")
                                
                                numeric_cols = get_numeric_columns(columns)
                                categorical_cols = [col for col in columns if col not in numeric_cols]
                                
                                return ui.markdown(f"""
                                **Numeric:** {len(numeric_cols)} columns  
                                **Categorical:** {len(categorical_cols)} columns  
                                **Total:** {len(columns)} columns
                                """)
                            except Exception as e:
                                logger.error(f"Error in dataset_stats: {e}")
                                return ui.markdown("**Total:** 0 columns")
                    
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("✅ Selection Summary")
                        @render.ui
                        def selection_summary():
                            try:
                                selected = selected_columns_store.get()
                                total = len(get_columns())
                                selected_count = len(selected) if selected else total
                                status = "All columns" if not selected else f"{selected_count} specific columns"
                                return ui.markdown(f"**Selected:** {status} ({selected_count}/{total})")
                            except Exception as e:
                                logger.error(f"Error in selection_summary: {e}")
                                return ui.markdown("**Selected:** 0/0 columns")
                    
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("🔍 Data Filter")
                        ui.input_text("column_search", "Search columns:", placeholder="Type to filter...")
                        ui.input_text("column_input", "Enter column names (semicolon-separated):", 
                                    placeholder="col1; col2; col3", value="")
                        ui.input_checkbox("show_complete_cases", "Use only complete cases (no missing values)", value=False)
                        with ui.layout_column_wrap(width=1/3, gap="0.5rem"):
                            ui.input_action_button("apply_selection", "Select", class_="btn-sm btn-primary")
                            ui.input_action_button("select_all", "Select All", class_="btn-sm btn-outline-primary")
                            ui.input_action_button("deselect_all", "Clear All", class_="btn-sm btn-outline-secondary")
                
                # Main column selection with search functionality
                @render.ui
                def columns_selector():
                    try:
                        columns = get_columns()
                        if not columns:
                            return ui.markdown("⚠️ No columns available in the dataset")
                        
                        search_term = input.column_search() or ""
                        
                        # Filter columns based on search
                        if search_term:
                            filtered_cols = [col for col in columns if search_term.lower() in col.lower()]
                        else:
                            filtered_cols = columns
                        
                        if not filtered_cols:
                            return ui.markdown("🔍 No columns match your search criteria")
                        
                        # Show available columns as a list
                        selected_cols = selected_columns_store.get()
                        available_cols_text = "\n".join([f"• {col}" for col in filtered_cols])
                        
                        if selected_cols:
                            selected_text = "\n".join([f"✅ {col}" for col in selected_cols if col in filtered_cols])
                            return ui.markdown(f"""
                            **Available columns:**\n{available_cols_text}\n\n
                            **Currently selected:**\n{selected_text}
                            """)
                        else:
                            return ui.markdown(f"""
                            **Available columns:**\n{available_cols_text}\n\n
                            **Currently selected:** All columns (✅ all shown above)
                            """)
                        
                    except Exception as e:
                        logger.error(f"Error in columns_selector: {e}")
                        return ui.markdown("❌ Error loading column selector")
                
                # Handle column selection functionality
                @reactive.Effect
                @reactive.event(input.apply_selection)
                def apply_column_selection():
                    try:
                        column_input = input.column_input() or ""
                        if not column_input.strip():
                            ui.notification_show("Please enter column names separated by semicolons", type="warning")
                            return
                        
                        # Parse semicolon-separated column names
                        input_cols = [col.strip() for col in column_input.split(';') if col.strip()]
                        all_cols = get_columns()
                        
                        # Validate that all input columns exist
                        valid_cols = [col for col in input_cols if col in all_cols]
                        invalid_cols = [col for col in input_cols if col not in all_cols]
                        
                        if invalid_cols:
                            ui.notification_show(f"Invalid columns: {', '.join(invalid_cols)}", type="warning")
                        
                        if valid_cols:
                            selected_columns_store.set(valid_cols)
                            ui.notification_show(f"Selected {len(valid_cols)} columns", type="success")
                        else:
                            ui.notification_show("No valid columns found", type="error")
                            
                    except Exception as e:
                        logger.error(f"Error applying column selection: {e}")
                        ui.notification_show(f"Error applying selection: {str(e)}", type="error")
                
                @reactive.Effect
                @reactive.event(input.select_all)
                def select_all_columns():
                    try:
                        columns = get_columns()
                        if columns:
                            selected_columns_store.set([])  # Empty list means all columns
                            ui.notification_show("All columns selected", type="success")
                    except Exception as e:
                        logger.error(f"Error selecting all columns: {e}")
                
                @reactive.Effect
                @reactive.event(input.deselect_all)
                def deselect_all_columns():
                    try:
                        selected_columns_store.set([])  # Empty list means all columns
                        ui.notification_show("All columns deselected (showing all)", type="success")
                    except Exception as e:
                        logger.error(f"Error deselecting all columns: {e}")
            
            @render.data_frame
            def data():
                try:
                    selected_cols = selected_columns_store.get()
                    df = safe_get_dataframe()
                    
                    if df.empty:
                        return pd.DataFrame({'Message': ['No data available']})
                    
                    if not selected_cols:
                        # If no specific columns selected, use all columns
                        display_df = df
                    else:
                        # Filter to only include columns that exist
                        valid_cols = [col for col in selected_cols if col in df.columns]
                        if not valid_cols:
                            return pd.DataFrame({'Message': ['No valid columns selected']})
                        display_df = df[valid_cols]
                    
                    # Apply complete cases filter if checkbox is checked
                    if input.show_complete_cases():
                        before_filter = len(display_df)
                        display_df = display_df.dropna()
                        after_filter = len(display_df)
                        if after_filter < before_filter:
                            ui.notification_show(f"Filtered from {before_filter} to {after_filter} complete cases", type="info")
                    
                    return display_df
                except Exception as e:
                    logger.error(f"Error in data display: {e}")
                    return pd.DataFrame({'Error': [f'Error loading data: {str(e)}']})
            
        with ui.nav_panel("Summary"):
            # Top row: Summary Statistics and Data Quality
            with ui.layout_column_wrap(width=1/3, gap="1rem"):
                # Dataset Overview Card
                with ui.card():
                    ui.h4("📋 Dataset Overview")
                    @render.ui
                    def dataset_overview():
                        try:
                            df = safe_get_dataframe()
                            if df.empty:
                                return ui.markdown("⚠️ No data available")
                            
                            selected_cols = selected_columns_store.get()
                            if not selected_cols:
                                selected_cols = list(df.columns)
                            
                            total_rows = len(df)
                            total_cols = len(selected_cols)
                            numeric_cols = len([col for col in selected_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])])
                            categorical_cols = total_cols - numeric_cols
                            
                            return ui.markdown(f"""
                            **Total Rows:** {total_rows:,}  
                            **Total Columns:** {total_cols}  
                            **Numeric Columns:** {numeric_cols}  
                            **Categorical Columns:** {categorical_cols}
                            """)
                        except Exception as e:
                            logger.error(f"Error in dataset_overview: {e}")
                            return ui.markdown("❌ Error loading dataset overview")
                
                # Data Quality Card
                with ui.card():
                    ui.h4("🔍 Data Quality")
                    @render.ui
                    def data_quality():
                        try:
                            df = safe_get_dataframe()
                            if df.empty:
                                return ui.markdown("⚠️ No data available")
                            
                            selected_cols = selected_columns_store.get()
                            if not selected_cols:
                                selected_cols = list(df.columns)
                            
                            valid_cols = [col for col in selected_cols if col in df.columns]
                            if not valid_cols:
                                return ui.markdown("⚠️ No valid columns selected")
                            
                            # Calculate data quality metrics
                            total_cells = len(df) * len(valid_cols)
                            missing_cells = df[valid_cols].isna().sum().sum()
                            missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
                            duplicate_rows = df.duplicated().sum()
                            duplicate_percentage = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0
                            
                            return ui.markdown(f"""
                            **Missing Values:** {missing_cells:,} ({missing_percentage:.1f}%)  
                            **Duplicate Rows:** {duplicate_rows:,} ({duplicate_percentage:.1f}%)  
                            **Complete Cases:** {len(df.dropna(subset=valid_cols)):,}
                            """)
                        except Exception as e:
                            logger.error(f"Error in data_quality: {e}")
                            return ui.markdown("❌ Error loading data quality metrics")
                
                # Quick Stats Card
                with ui.card():
                    ui.h4("📊 Quick Statistics")
                    @render.ui
                    def quick_stats():
                        try:
                            df = safe_get_dataframe()
                            if df.empty:
                                return ui.markdown("⚠️ No data available")
                            
                            selected_cols = selected_columns_store.get()
                            if not selected_cols:
                                selected_cols = list(df.columns)
                            
                            numeric_cols = [col for col in selected_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                            if not numeric_cols:
                                return ui.markdown("⚠️ No numeric columns available")
                            
                            # Calculate summary statistics
                            numeric_data = df[numeric_cols].dropna()
                            if numeric_data.empty:
                                return ui.markdown("⚠️ No numeric data available")
                            
                            mean_vals = numeric_data.mean()
                            std_vals = numeric_data.std()
                            min_vals = numeric_data.min()
                            max_vals = numeric_data.max()
                            
                            # Find columns with highest/lowest values
                            highest_mean_col = mean_vals.idxmax()
                            highest_mean_val = mean_vals.max()
                            lowest_mean_col = mean_vals.idxmin()
                            lowest_mean_val = mean_vals.min()
                            
                            return ui.markdown(f"""
                            **Highest Mean:** {highest_mean_col} ({highest_mean_val:.2f})  
                            **Lowest Mean:** {lowest_mean_col} ({lowest_mean_val:.2f})  
                            **Variables Analyzed:** {len(numeric_cols)}
                            """)
                        except Exception as e:
                            logger.error(f"Error in quick_stats: {e}")
                            return ui.markdown("❌ Error loading quick statistics")
            
            # Middle row: Distribution and Correlation Analysis
            with ui.layout_column_wrap(width=1/2, gap="1rem"):
                # Distribution Analysis Card
                with ui.card():
                    ui.h4("📊 Distribution Analysis")
                    ui.markdown("Explore the distribution of numeric variables with histograms and density plots.")
                    
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("📈 Variable Selection")
                        @render.ui
                        def var_selector():
                            try:
                                numeric_cols = get_numeric_columns()
                                if not numeric_cols:
                                    return ui.markdown("⚠️ No numeric columns available for plotting")
                                
                                return ui.input_select(
                                    "var",
                                    "Select variable:",
                                    choices=numeric_cols,
                                    selected=numeric_cols[0] if numeric_cols else None
                                )
                            except Exception as e:
                                logger.error(f"Error in var_selector: {e}")
                                return ui.markdown("❌ Error loading variable selector")
                    
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("⚙️ Plot Settings")
                        ui.input_text('bins', 'Number of bins:', value="50")
                        ui.input_text('bw_adjust', 'Bandwidth adjustment:', value="1.0")
                    
                    @render.plot
                    def hist():
                        try:
                            var = input.var()
                            if not var:
                                return None
                            
                            bins = validate_numeric_input(input.bins(), 50, min_value=1, max_value=1000)
                            bw = validate_numeric_input(input.bw_adjust(), 1.0, min_value=0.1, max_value=10.0)
                            
                            df = safe_get_dataframe()
                            if df.empty or var not in df.columns:
                                return None
                            
                            # Check for numeric data
                            if not pd.api.types.is_numeric_dtype(df[var]):
                                ui.notification_show(f"Column '{var}' is not numeric. Cannot create histogram.", type="warning")
                                return None
                            
                            # Remove NA values for plotting
                            plot_data = df[var].dropna()
                            if plot_data.empty:
                                ui.notification_show(f"No valid numeric data in column '{var}'", type="warning")
                                return None
                            
                            p = sns.histplot(
                                plot_data, 
                                stat="density",
                                facecolor=CUSTOM_COLORS['secondary'], 
                                edgecolor="#ebd9b2",
                                alpha=0.8,
                                bins=int(bins),
                            )
                            sns.kdeplot(
                                plot_data, 
                                color=CUSTOM_COLORS['primary'], 
                                linewidth=2,
                                bw_adjust=bw
                            )
                            return p.set(xlabel=var)
                            
                        except Exception as e:
                            logger.error(f"Error in histogram: {e}")
                            ui.notification_show(f"Error creating histogram: {str(e)}", type="error")
                            return None
                
                # Correlation Analysis Card
                with ui.card():
                    ui.h4("🔗 Correlation Analysis")
                    ui.markdown("Explore relationships between numeric variables with correlation heatmaps.")
                    
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("📊 Variable Selection")
                        @render.ui
                        def corr_selector():
                            try:
                                numeric_cols = get_numeric_columns()
                                if len(numeric_cols) < 2:
                                    return ui.markdown("⚠️ Need at least 2 numeric columns for correlation analysis")
                                
                                return ui.input_checkbox_group(
                                    "selected_numerical",
                                    "Select columns to display:",
                                    choices=numeric_cols,
                                    selected=numeric_cols,
                                )
                            except Exception as e:
                                logger.error(f"Error in corr_selector: {e}")
                                return ui.markdown("❌ Error loading correlation selector")
                    
                    with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                        ui.h5("📈 Analysis Info")
                        @render.ui
                        def corr_info():
                            try:
                                selected_cols = list(input.selected_numerical())
                                if len(selected_cols) < 2:
                                    return ui.markdown("Select at least 2 variables")
                                
                                df = safe_get_dataframe()
                                if df.empty:
                                    return ui.markdown("No data available")
                                
                                valid_cols = [col for col in selected_cols if col in df.columns]
                                if len(valid_cols) < 2:
                                    return ui.markdown("Invalid column selection")
                                
                                corr_data = df[valid_cols].dropna()
                                return ui.markdown(f"""
                                **Variables:** {len(valid_cols)} selected  
                                **Observations:** {len(corr_data)} complete cases  
                                **Matrix size:** {len(valid_cols)} × {len(valid_cols)}
                                """)
                            except Exception as e:
                                logger.error(f"Error in corr_info: {e}")
                                return ui.markdown("Error loading correlation info")
                    
                    @render_widget
                    def corr():
                        try:
                            selected_cols = list(input.selected_numerical())
                            if len(selected_cols) < 2:
                                return None
                            
                            df = safe_get_dataframe()
                            if df.empty:
                                return None
                            
                            # Filter to valid columns
                            valid_cols = [col for col in selected_cols if col in df.columns]
                            if len(valid_cols) < 2:
                                return None
                            
                            # Remove rows with any NA values for correlation
                            corr_data = df[valid_cols].dropna()
                            if corr_data.empty:
                                ui.notification_show("No complete data available for correlation analysis", type="warning")
                                return None
                            
                            corr_matrix = corr_data.corr()
                            
                            # Check for valid correlation matrix
                            if corr_matrix.isna().all().all():
                                ui.notification_show("Unable to calculate correlations (all values are NA)", type="warning")
                                return None
                            
                            p = px.imshow(
                                corr_matrix, 
                                title="Correlation Heatmap",
                                color_continuous_scale='YlOrRd', 
                                labels={'x': 'Features', 'y': 'Features'}, 
                                width=600, height=600
                            )
                            return apply_plotly_styling(p, title="Correlation Heatmap")
                            
                        except Exception as e:
                            logger.error(f"Error in correlation plot: {e}")
                            ui.notification_show(f"Error creating correlation plot: {str(e)}", type="error")
                            return None
            
            # Bottom row: Additional Summary Visualizations
            with ui.layout_column_wrap(width=1/2, gap="1rem"):
                # Missing Values Heatmap Card
                with ui.card():
                    ui.h4("❓ Missing Values Pattern")
                    ui.markdown("Visualize the pattern of missing values across your dataset.")
                    
                    @render.ui
                    def missing_heatmap():
                        try:
                            df = safe_get_dataframe()
                            if df.empty:
                                return ui.markdown("⚠️ No data available")
                            
                            selected_cols = selected_columns_store.get()
                            if not selected_cols:
                                selected_cols = list(df.columns)
                            
                            valid_cols = [col for col in selected_cols if col in df.columns]
                            if not valid_cols:
                                return ui.markdown("⚠️ No valid columns selected")
                            
                            # Create missing values heatmap
                            missing_data = df[valid_cols].isna()
                            if not missing_data.any().any():
                                return ui.markdown("✅ No missing values found in the dataset!")
                            
                            # Sample data for visualization (first 100 rows)
                            sample_size = min(100, len(missing_data))
                            sample_missing = missing_data.head(sample_size)
                            
                            p = px.imshow(
                                sample_missing.T,  # Transpose to show variables on y-axis
                                title=f"Missing Values Pattern (First {sample_size} rows)",
                                color_continuous_scale='Reds',
                                labels={'x': 'Row Index', 'y': 'Variables'},
                                width=600, height=400
                            )
                            p.update_layout(
                                xaxis_title="Row Index",
                                yaxis_title="Variables",
                                yaxis={'tickmode': 'array', 'tickvals': list(range(len(valid_cols))), 'ticktext': valid_cols}
                            )
                            styled_p = apply_plotly_styling(p, title="Missing Values Pattern")
                            return ui.HTML(styled_p.to_html())
                            
                        except Exception as e:
                            logger.error(f"Error in missing heatmap: {e}")
                            return ui.markdown("❌ Error creating missing values heatmap")
                
                # Summary Statistics Table Card
                with ui.card():
                    ui.h4("📈 Summary Statistics")
                    ui.markdown("Detailed statistical summary of numeric variables.")
                    
                    @render.data_frame
                    def summary_stats_table():
                        try:
                            df = safe_get_dataframe()
                            if df.empty:
                                return pd.DataFrame({'Message': ['No data available']})
                            
                            selected_cols = selected_columns_store.get()
                            if not selected_cols:
                                selected_cols = list(df.columns)
                            
                            numeric_cols = [col for col in selected_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                            if not numeric_cols:
                                return pd.DataFrame({'Message': ['No numeric columns available']})
                            
                            # Calculate summary statistics
                            summary_stats = df[numeric_cols].describe()
                            
                            # Add additional statistics
                            summary_stats.loc['missing'] = df[numeric_cols].isna().sum()
                            summary_stats.loc['missing_pct'] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)
                            summary_stats.loc['unique'] = df[numeric_cols].nunique()
                            
                            # Round numeric values
                            summary_stats = summary_stats.round(2)
                            
                            # Rename index with descriptive labels
                            summary_stats = summary_stats.rename(index={
                                'count': 'Count',
                                'mean': 'Mean',
                                'std': 'Std Dev',
                                'min': 'Min',
                                '25%': '25th %',
                                '50%': 'Median',
                                '75%': '75th %',
                                'max': 'Max',
                                'missing': 'Missing',
                                'missing_pct': 'Missing %',
                                'unique': 'Unique'
                            })
                            
                            # Reset index to make statistic names a column
                            summary_stats = summary_stats.reset_index()
                            summary_stats = summary_stats.rename(columns={'index': 'Statistic'})
                            
                            return summary_stats
                            
                        except Exception as e:
                            logger.error(f"Error in summary stats table: {e}")
                            return pd.DataFrame({'Error': [f'Error creating summary statistics: {str(e)}']})
            
            # New row: 2D Regression Plot
            with ui.layout_column_wrap(width=1/1, gap="1rem"):
                # 2D Regression Plot Card
                with ui.card():
                    ui.h4("📊 2D Regression Analysis")
                    ui.markdown("Explore the relationship between two numeric variables with regression analysis.")
                    
                    with ui.layout_column_wrap(width=1/2, gap="1rem"):
                        with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                            ui.h5("📈 Variable Selection")
                            @render.ui
                            def x_var_selector():
                                try:
                                    numeric_cols = get_numeric_columns()
                                    if not numeric_cols:
                                        return ui.markdown("⚠️ No numeric columns available")
                                    
                                    return ui.input_select(
                                        "x_var",
                                        "Select X-axis variable:",
                                        choices=numeric_cols,
                                        selected=numeric_cols[0] if numeric_cols else None
                                    )
                                except Exception as e:
                                    logger.error(f"Error in x_var_selector: {e}")
                                    return ui.markdown("❌ Error loading X variable selector")
                            
                            @render.ui
                            def y_var_selector():
                                try:
                                    numeric_cols = get_numeric_columns()
                                    if not numeric_cols:
                                        return ui.markdown("⚠️ No numeric columns available")
                                    
                                    selected_idx = 1 if len(numeric_cols) > 1 else 0
                                    return ui.input_select(
                                        "y_var",
                                        "Select Y-axis variable:",
                                        choices=numeric_cols,
                                        selected=numeric_cols[selected_idx] if numeric_cols else None
                                    )
                                except Exception as e:
                                    logger.error(f"Error in y_var_selector: {e}")
                                    return ui.markdown("❌ Error loading Y variable selector")
                        
                        with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                            ui.h5("📊 Plot Info")
                            @render.ui
                            def regression_info():
                                try:
                                    x_var = input.x_var()
                                    y_var = input.y_var()
                                    
                                    if not x_var or not y_var:
                                        return ui.markdown("Select X and Y variables")
                                    
                                    df = safe_get_dataframe()
                                    if df.empty:
                                        return ui.markdown("No data available")
                                    
                                    if x_var not in df.columns or y_var not in df.columns:
                                        return ui.markdown("Invalid variable selection")
                                    
                                    plot_data = df[[x_var, y_var]].dropna()
                                    if plot_data.empty:
                                        return ui.markdown("No complete data available")
                                    
                                    return ui.markdown(f"""
                                    **X Variable:** {x_var}  
                                    **Y Variable:** {y_var}  
                                    **Observations:** {len(plot_data)} complete cases  
                                    **Correlation:** {plot_data[x_var].corr(plot_data[y_var]):.3f}
                                    """)
                                except Exception as e:
                                    logger.error(f"Error in regression_info: {e}")
                                    return ui.markdown("Error loading regression info")
                    
                    @render_widget
                    def regression():
                        try:
                            x_var = input.x_var()
                            y_var = input.y_var()
                            
                            if not x_var or not y_var:
                                return None
                            
                            df = safe_get_dataframe()
                            if df.empty:
                                return None
                            
                            # Check if columns exist
                            if x_var not in df.columns or y_var not in df.columns:
                                ui.notification_show("Selected variables not found in dataset", type="error")
                                return None
                            
                            # Remove NA values for regression
                            plot_data = df[[x_var, y_var]].dropna()
                            if plot_data.empty:
                                ui.notification_show("No complete data available for regression", type="warning")
                                return None
                            
                            if len(plot_data) < 3:
                                ui.notification_show("Insufficient data points for regression analysis", type="warning")
                                return None
                            
                            p = px.scatter(
                                x=plot_data[x_var],
                                y=plot_data[y_var],
                                trendline='ols',
                                trendline_color_override=CUSTOM_COLORS['accent'],
                                title='Regression Plot'
                            )
                            p.update_traces(
                                marker=dict(
                                    size=8, 
                                    opacity=0.8, 
                                    color=CUSTOM_COLORS['secondary'], 
                                    line=dict(width=1, color="black")
                                )
                            )
                            
                            p = apply_plotly_styling(p, title='Regression Plot', x_title=x_var, y_title=y_var)
                            
                            # Add mean annotation
                            try:
                                p.add_annotation(
                                    x=plot_data[x_var].mean(), 
                                    y=plot_data[y_var].mean(),
                                    text="Mean",
                                    showarrow=True, 
                                    arrowhead=2, 
                                    arrowcolor="black",
                                    font=dict(size=14, color="black")
                                )
                            except Exception as e:
                                logger.warning(f"Error adding mean annotation: {e}")
                            
                            return p
                            
                        except Exception as e:
                            logger.error(f"Error in regression plot: {e}")
                            ui.notification_show(f"Error creating regression plot: {str(e)}", type="error")
                            return None

# Panel Data Panel
with ui.nav_panel("Panel Data"):
    # Top row: Panel Data Overview and Settings
    with ui.layout_column_wrap(width=1/3, gap="1rem"):
        # Panel Data Overview Card
        with ui.card():
            ui.h4("📊 Panel Data Overview")
            @render.ui
            def panel_data_overview():
                try:
                    df = safe_get_dataframe()
                    if df.empty:
                        return ui.markdown("⚠️ No data available")
                    
                    year_cols = get_year_columns()
                    numeric_cols = get_numeric_columns()
                    
                    total_rows = len(df)
                    year_cols_count = len(year_cols)
                    numeric_cols_count = len(numeric_cols)
                    
                    return ui.markdown(f"""
                    **Total Observations:** {total_rows:,}  
                    **Date/Time Columns:** {year_cols_count}  
                    **Numeric Variables:** {numeric_cols_count}  
                    **Available for Analysis:** {min(year_cols_count, numeric_cols_count)} combinations
                    """)
                except Exception as e:
                    logger.error(f"Error in panel_data_overview: {e}")
                    return ui.markdown("❌ Error loading panel data overview")
    
        # Data Processing Info Card
        with ui.card():
            ui.h4("⚙️ Processing Info")
            @render.ui
            def processing_info():
                try:
                    year_col = input.year_col() if hasattr(input, 'year_col') else None
                    y_col = input.y_time_var() if hasattr(input, 'y_time_var') else None
                    aggregation_option = input.aggregation_option() if hasattr(input, 'aggregation_option') else "none"
                    
                    if not year_col or not y_col:
                        return ui.markdown("⚠️ Please select date column and Y variable")
                    
                    df = safe_get_dataframe()
                    if df.empty:
                        return ui.markdown("⚠️ No data available")
                    
                    # Calculate processing info
                    total_obs = len(df)
                    missing_obs = df[[year_col, y_col]].isna().any(axis=1).sum()
                    complete_obs = total_obs - missing_obs
                    
                    agg_text = "None" if aggregation_option == "none" else aggregation_option.replace("_", " ").title()
                    
                    return ui.markdown(f"""
                    **Total Observations:** {total_obs:,}  
                    **Complete Cases:** {complete_obs:,}  
                    **Missing Cases:** {missing_obs:,}  
                    **Aggregation:** {agg_text}
                    """)
                except Exception as e:
                    logger.error(f"Error in processing_info: {e}")
                    return ui.markdown("❌ Error loading processing info")
    
        # Variable Selection Status Card
        with ui.card():
            ui.h4("🎯 Selection Status")
            @render.ui
            def selection_status():
                try:
                    year_col = input.year_col() if hasattr(input, 'year_col') else None
                    y_col = input.y_time_var() if hasattr(input, 'y_time_var') else None
                    numerator_col = input.numerator_var() if hasattr(input, 'numerator_var') else None
                    denominator_col = input.denominator_var() if hasattr(input, 'denominator_var') else None
                    
                    simple_ready = year_col and y_col
                    ratio_ready = year_col and numerator_col and denominator_col
                    
                    return ui.markdown(f"""
                    **Simple Plot:** {'✅ Ready' if simple_ready else '⚠️ Incomplete'}  
                    **Ratio Plot:** {'✅ Ready' if ratio_ready else '⚠️ Incomplete'}  
                    **Date Column:** {year_col or 'Not selected'}  
                    **Y Variable:** {y_col or 'Not selected'}
                    """)
                except Exception as e:
                    logger.error(f"Error in selection_status: {e}")
                    return ui.markdown("❌ Error loading selection status")
    
    # Middle row: Settings and Configuration
    with ui.layout_column_wrap(width=1/2, gap="1rem"):
        # Simple Plot Settings Card
        with ui.card():
            ui.h4("📈 Simple Plot Settings")
            ui.markdown("Configure variables and options for simple time series analysis.")
            
            with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                ui.h5("📅 Variable Selection")
                @render.ui
                def year_col_selector():
                    try:
                        year_cols = get_year_columns()
                        if not year_cols:
                            return ui.markdown("⚠️ No suitable date/year columns found")
                        
                        return ui.input_select(
                            "year_col",
                            "Select date column:",
                            choices=year_cols,
                            selected=year_cols[0] if year_cols else None
                        )
                    except Exception as e:
                        logger.error(f"Error in year_col_selector: {e}")
                        return ui.markdown("❌ Error loading year column selector")
                
                @render.ui
                def y_time_var_selector():
                    try:
                        numeric_cols = get_numeric_columns()
                        if not numeric_cols:
                            return ui.markdown("⚠️ No numeric columns available")
                        
                        return ui.input_select(
                            "y_time_var",
                            "Select Y variable:",
                            choices=numeric_cols,
                            selected=numeric_cols[0] if numeric_cols else None
                        )
                    except Exception as e:
                        logger.error(f"Error in y_time_var_selector: {e}")
                        return ui.markdown("❌ Error loading Y variable selector")
            
            with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                ui.h5("⚙️ Processing Options")
                ui.input_radio_buttons(
                    "aggregation_option",
                    "Aggregation option:",
                    choices={
                        "none": "None",
                        "by_date": "Aggregate by selected date column",
                        "by_year_sum": "Aggregate to year level (sum)",
                        "by_year_mean": "Aggregate to year level (mean)"
                    },
                    selected="none",
                    inline=True
                )
                
                @render.ui
                def filter_var_ui():
                    try:
                        year_col = input.year_col()
                        y_col = input.y_time_var()
                        filter_choices = ["None"] + [col for col in get_selected_columns() if col != year_col and col != y_col]
                        return ui.input_select(
                            "filter_var",
                            "Filter variable:",
                            choices=filter_choices,
                            selected="None"
                        )
                    except Exception as e:
                        logger.error(f"Error in filter_var_ui: {e}")
                        return ui.markdown("❌ Error loading filter variable selector")

                @render.ui
                def filter_value_ui():
                    try:
                        var = input.filter_var()
                        if var and var != "None":
                            df = safe_get_dataframe()
                            if df.empty or var not in df.columns:
                                return None
                            
                            values = sorted(df[var].dropna().unique())
                            if not values:
                                return ui.markdown("⚠️ No values available for filtering")
                            
                            return ui.input_select(
                                "filter_value",
                                f"Select value for {var}:",
                                choices=[str(v) for v in values],
                                selected=None
                            )
                        return None
                    except Exception as e:
                        logger.error(f"Error in filter_value_ui: {e}")
                        return ui.markdown("❌ Error loading filter value selector")
        
        # Ratio Plot Settings Card
        with ui.card():
            ui.h4("📊 Ratio Plot Settings")
            ui.markdown("Configure variables and options for ratio-based time series analysis.")
            
            with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                ui.h5("📈 Variable Selection")
                @render.ui
                def numerator_var_selector():
                    try:
                        numeric_cols = get_numeric_columns()
                        if not numeric_cols:
                            return ui.markdown("⚠️ No numeric columns available")
                        
                        return ui.input_select(
                            "numerator_var",
                            "Select numerator variable:",
                            choices=numeric_cols,
                            selected=numeric_cols[0] if numeric_cols else None
                        )
                    except Exception as e:
                        logger.error(f"Error in numerator_var_selector: {e}")
                        return ui.markdown("❌ Error loading numerator variable selector")
                
                @render.ui
                def denominator_var_selector():
                    try:
                        numeric_cols = get_numeric_columns()
                        if not numeric_cols:
                            return ui.markdown("⚠️ No numeric columns available")
                        
                        selected_idx = 1 if len(numeric_cols) > 1 else 0
                        return ui.input_select(
                            "denominator_var",
                            "Select denominator variable:",
                            choices=numeric_cols,
                            selected=numeric_cols[selected_idx] if numeric_cols else None
                        )
                    except Exception as e:
                        logger.error(f"Error in denominator_var_selector: {e}")
                        return ui.markdown("❌ Error loading denominator variable selector")
            
            with ui.card(style="background-color: #f8f9fa; border: 1px solid #dee2e6;"):
                ui.h5("⚙️ Processing Options")
                ui.input_radio_buttons(
                    "ratio_aggregation_option",
                    "Aggregation option:",
                    choices={
                        "none": "None",
                        "by_date": "Aggregate by selected date column",
                        "by_year_sum": "Aggregate to year level (sum)",
                        "by_year_mean": "Aggregate to year level (mean)"
                    },
                    selected="none",
                    inline=True
                )
                
                @render.ui
                def ratio_filter_var_ui():
                    try:
                        year_col = input.year_col()
                        numerator_col = input.numerator_var()
                        denominator_col = input.denominator_var()
                        filter_choices = ["None"] + [col for col in get_selected_columns() if col != year_col and col != numerator_col and col != denominator_col]
                        return ui.input_select(
                            "ratio_filter_var",
                            "Filter variable:",
                            choices=filter_choices,
                            selected="None"
                        )
                    except Exception as e:
                        logger.error(f"Error in ratio_filter_var_ui: {e}")
                        return ui.markdown("❌ Error loading ratio filter variable selector")

                @render.ui
                def ratio_filter_value_ui():
                    try:
                        var = input.ratio_filter_var()
                        if var and var != "None":
                            df = safe_get_dataframe()
                            if df.empty or var not in df.columns:
                                return None
                            
                            values = sorted(df[var].dropna().unique())
                            if not values:
                                return ui.markdown("⚠️ No values available for filtering")
                            
                            return ui.input_select(
                                "ratio_filter_value",
                                f"Select value for {var}:",
                                choices=[str(v) for v in values],
                                selected=None
                            )
                        return None
                    except Exception as e:
                        logger.error(f"Error in ratio_filter_value_ui: {e}")
                        return ui.markdown("❌ Error loading ratio filter value selector")
    
    # Bottom row: Time Series Charts
    with ui.layout_column_wrap(width=1/2, gap="1rem"):
        # Simple Plot Card
        with ui.card():
            ui.h4("📈 Simple Time Series Plot")
            ui.markdown("Visualize the time series of your selected variable.")
            
            @render_widget
            def time_series_chart():
                try:
                    year_col = input.year_col()
                    y_col = input.y_time_var()
                    aggregation_option = input.aggregation_option()
                    filter_var = input.filter_var() if hasattr(input, 'filter_var') else "None"
                    filter_value = input.filter_value() if (hasattr(input, 'filter_value') and filter_var not in (None, "None", "")) else None
                    
                    if not year_col or not y_col:
                        return None
                    
                    dff, n_obs, n_dropped, group_col, y_col = process_panel_data(
                        year_col, y_col, aggregation_option, filter_var, filter_value
                    )
                    
                    if dff.empty:
                        ui.notification_show("No data available for plotting after processing", type="warning")
                        return None
                    
                    title = f"Time Series of {y_col} by {group_col}"
                    if aggregation_option != "none":
                        title += " (Aggregated)"
                    
                    p = px.line(
                        dff,
                        x=group_col,
                        y=y_col,
                        markers=True,
                        title=title
                    )
                    p.update_traces(
                        line=dict(color=CUSTOM_COLORS['primary'], width=3), 
                        marker=dict(size=8, color=CUSTOM_COLORS['secondary'])
                    )
                    
                    return apply_plotly_styling(p, title=title, x_title=group_col, y_title=y_col)
                    
                except Exception as e:
                    logger.error(f"Error in time_series_chart: {e}")
                    ui.notification_show(f"Error creating time series chart: {str(e)}", type="error")
                    return None
            
            @render.ui
            def n_obs_text():
                try:
                    year_col = input.year_col()
                    y_col = input.y_time_var()
                    aggregation_option = input.aggregation_option()
                    filter_var = input.filter_var() if hasattr(input, 'filter_var') else "None"
                    filter_value = input.filter_value() if (hasattr(input, 'filter_value') and filter_var not in (None, "None", "")) else None
                    
                    if not year_col or not y_col:
                        return ui.markdown("⚠️ Please select both year column and Y variable")
                    
                    _, n_obs, n_dropped, _, _ = process_panel_data(
                        year_col, y_col, aggregation_option, filter_var, filter_value
                    )
                    return ui.markdown(f"**Number of observations:** {n_obs} &nbsp;&nbsp; | &nbsp;&nbsp; **Dropped:** {n_dropped}")
                    
                except Exception as e:
                    logger.error(f"Error in n_obs_text: {e}")
                    return ui.markdown("❌ Error calculating observations")

        # Ratio Plot Card
        with ui.card():
            ui.h4("📊 Ratio Time Series Plot")
            ui.markdown("Visualize the ratio of two variables over time.")
            
            @render_widget
            def ratio_time_series_chart():
                try:
                    year_col = input.year_col()
                    numerator_col = input.numerator_var()
                    denominator_col = input.denominator_var()
                    aggregation_option = input.ratio_aggregation_option()
                    filter_var = input.ratio_filter_var() if hasattr(input, 'ratio_filter_var') else "None"
                    filter_value = input.ratio_filter_value() if (hasattr(input, 'ratio_filter_value') and filter_var not in (None, "None", "")) else None
                    
                    if not year_col or not numerator_col or not denominator_col:
                        return None
                    
                    dff, n_obs, n_dropped, group_col, ratio_col = process_ratio_data(
                        year_col, numerator_col, denominator_col, aggregation_option, filter_var, filter_value
                    )
                    
                    if dff.empty:
                        ui.notification_show("No data available for ratio plotting after processing", type="warning")
                        return None
                    
                    title = f"Ratio Plot: {numerator_col} / {denominator_col} by {group_col}"
                    if aggregation_option != "none":
                        title += " (Aggregated)"
                    
                    p = px.line(
                        dff,
                        x=group_col,
                        y=ratio_col,
                        markers=True,
                        title=title
                    )
                    p.update_traces(
                        line=dict(color=CUSTOM_COLORS['accent'], width=3), 
                        marker=dict(size=8, color=CUSTOM_COLORS['secondary'])
                    )
                    
                    return apply_plotly_styling(
                        p, 
                        title=title, 
                        x_title=group_col, 
                        y_title=f"{numerator_col} / {denominator_col}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in ratio_time_series_chart: {e}")
                    ui.notification_show(f"Error creating ratio time series chart: {str(e)}", type="error")
                    return None
            
            @render.ui
            def ratio_n_obs_text():
                try:
                    year_col = input.year_col()
                    numerator_col = input.numerator_var()
                    denominator_col = input.denominator_var()
                    aggregation_option = input.ratio_aggregation_option()
                    filter_var = input.ratio_filter_var() if hasattr(input, 'ratio_filter_var') else "None"
                    filter_value = input.ratio_filter_value() if (hasattr(input, 'ratio_filter_value') and filter_var not in (None, "None", "")) else None
                    
                    if not year_col or not numerator_col or not denominator_col:
                        return ui.markdown("⚠️ Please select year column, numerator, and denominator variables")
                    
                    _, n_obs, n_dropped, _, _ = process_ratio_data(
                        year_col, numerator_col, denominator_col, aggregation_option, filter_var, filter_value
                    )
                    return ui.markdown(f"**Number of observations:** {n_obs} &nbsp;&nbsp; | &nbsp;&nbsp; **Dropped:** {n_dropped}")
                    
                except Exception as e:
                    logger.error(f"Error in ratio_n_obs_text: {e}")
                    return ui.markdown("❌ Error calculating ratio observations")

# Multiple Regression Panel
with ui.nav_panel("Multiple Regression"):
    with ui.navset_card_underline(title="Multiple Regression"):
        with ui.nav_panel("Settings"):
            @render.ui
            def dep_var_selector():
                try:
                    numeric_cols = get_numeric_columns()
                    if not numeric_cols:
                        return ui.markdown("⚠️ No numeric columns available")
                    
                    return ui.input_select(
                        'dep_var',
                        'Select dependent variable:',
                        choices=numeric_cols
                    )
                except Exception as e:
                    logger.error(f"Error in dep_var_selector: {e}")
                    return ui.markdown("❌ Error loading dependent variable selector")
            
            @render.ui
            def dynamic_indep_vars():
                try:
                    dep_var = input.dep_var()
                    choices = [x for x in get_selected_columns() if x != dep_var]
                    
                    if not choices:
                        return ui.markdown("⚠️ No independent variables available")
                    
                    return ui.input_checkbox_group(
                        "indep_var",
                        "Select independent variables:",
                        choices=choices,
                        selected=choices,
                    )
                except Exception as e:
                    logger.error(f"Error in dynamic_indep_vars: {e}")
                    return ui.markdown("❌ Error loading independent variables selector")
            
            ui.input_select(
                'covtype',
                'Select covariance type:',
                choices=['standard', 'white', 'hc1', 'hc2']
            )
            ui.input_checkbox(
                'constant',
                'Include constant',
                value=True
            )
        
        with ui.nav_panel("Results"):
            @render.data_frame
            def results():
                try:
                    if not input.indep_var():
                        return pd.DataFrame({'Message': ['Please select at least one independent variable']})
                    
                    dep_var = input.dep_var()
                    indep_vars = list(input.indep_var())
                    
                    if not dep_var:
                        return pd.DataFrame({'Message': ['Please select a dependent variable']})
                    
                    df = safe_get_dataframe()
                    if df.empty:
                        return pd.DataFrame({'Message': ['No data available']})
                    
                    # Check if all variables exist
                    all_vars = [dep_var] + indep_vars
                    missing_vars = [var for var in all_vars if var not in df.columns]
                    if missing_vars:
                        return pd.DataFrame({'Error': [f'Variables not found: {", ".join(missing_vars)}']})
                    
                    # Remove rows with any NA values
                    model_data = df[all_vars].dropna()
                    if model_data.empty:
                        return pd.DataFrame({'Error': ['No complete data available for regression']})
                    
                    if len(model_data) < len(indep_vars) + 2:
                        return pd.DataFrame({'Error': ['Insufficient data points for regression analysis']})
                    
                    model = OLS(
                        model_data[dep_var],
                        model_data[indep_vars]
                    )
                    model_store.set(model)
                    table = model.fit(covtype=input.covtype(), constant=input.constant())
                    return table
                    
                except Exception as e:
                    logger.error(f"Error in regression results: {e}")
                    return pd.DataFrame({'Error': [f'Error in regression analysis: {str(e)}']})
        
        with ui.nav_panel("Diagnostics"):
            @render_widget
            def residual_plot():
                try:
                    if model_store.get() is None:
                        return None
                        
                    model = model_store.get()
                    
                    # Check if model has fitted values and residuals
                    if not hasattr(model, 'fitted') or not hasattr(model, 'residuals'):
                        ui.notification_show("Model diagnostics not available", type="warning")
                        return None
                    
                    if len(model.fitted) == 0 or len(model.residuals) == 0:
                        ui.notification_show("No fitted values or residuals available", type="warning")
                        return None
                    
                    p = px.scatter(
                        x=model.fitted,
                        y=model.residuals,
                        labels={'x': 'Fitted values', 'y': 'Residuals'},
                        title='Residual Plot'
                    )
                    p.add_hline(y=0, line_dash="dash", line_color="red")
                    p.update_traces(
                        marker=dict(
                            size=8, 
                            opacity=0.8, 
                            color=CUSTOM_COLORS['secondary'], 
                            line=dict(width=1, color="black")
                        )
                    )
                    
                    return apply_plotly_styling(p, title='Residual Plot')
                    
                except Exception as e:
                    logger.error(f"Error in residual plot: {e}")
                    ui.notification_show(f"Error creating residual plot: {str(e)}", type="error")
                    return None

# Input tracking reactive effects
# These effects track all input values as they change, ensuring we can capture the complete state
# regardless of which panel is currently open

@reactive.Effect
@reactive.event(input.var)
def track_var():
    """Track var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['var'] = input.var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track var input: {e}")

@reactive.Effect
@reactive.event(input.bins)
def track_bins():
    """Track bins input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['bins'] = input.bins()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track bins input: {e}")

@reactive.Effect
@reactive.event(input.bw_adjust)
def track_bw_adjust():
    """Track bw_adjust input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['bw_adjust'] = input.bw_adjust()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track bw_adjust input: {e}")

@reactive.Effect
@reactive.event(input.selected_numerical)
def track_selected_numerical():
    """Track selected_numerical input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['selected_numerical'] = input.selected_numerical()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track selected_numerical input: {e}")

@reactive.Effect
@reactive.event(input.year_col)
def track_year_col():
    """Track year_col input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['year_col'] = input.year_col()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track year_col input: {e}")

@reactive.Effect
@reactive.event(input.y_time_var)
def track_y_time_var():
    """Track y_time_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['y_time_var'] = input.y_time_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track y_time_var input: {e}")

@reactive.Effect
@reactive.event(input.aggregation_option)
def track_aggregation_option():
    """Track aggregation_option input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['aggregation_option'] = input.aggregation_option()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track aggregation_option input: {e}")

@reactive.Effect
@reactive.event(input.filter_var)
def track_filter_var():
    """Track filter_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['filter_var'] = input.filter_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track filter_var input: {e}")

@reactive.Effect
@reactive.event(input.filter_value)
def track_filter_value():
    """Track filter_value input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['filter_value'] = input.filter_value()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track filter_value input: {e}")

@reactive.Effect
@reactive.event(input.numerator_var)
def track_numerator_var():
    """Track numerator_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['numerator_var'] = input.numerator_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track numerator_var input: {e}")

@reactive.Effect
@reactive.event(input.denominator_var)
def track_denominator_var():
    """Track denominator_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['denominator_var'] = input.denominator_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track denominator_var input: {e}")

@reactive.Effect
@reactive.event(input.ratio_aggregation_option)
def track_ratio_aggregation_option():
    """Track ratio_aggregation_option input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['ratio_aggregation_option'] = input.ratio_aggregation_option()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track ratio_aggregation_option input: {e}")

@reactive.Effect
@reactive.event(input.ratio_filter_var)
def track_ratio_filter_var():
    """Track ratio_filter_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['ratio_filter_var'] = input.ratio_filter_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track ratio_filter_var input: {e}")

@reactive.Effect
@reactive.event(input.ratio_filter_value)
def track_ratio_filter_value():
    """Track ratio_filter_value input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['ratio_filter_value'] = input.ratio_filter_value()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track ratio_filter_value input: {e}")

@reactive.Effect
@reactive.event(input.x_var)
def track_x_var():
    """Track x_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['x_var'] = input.x_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track x_var input: {e}")

@reactive.Effect
@reactive.event(input.y_var)
def track_y_var():
    """Track y_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['y_var'] = input.y_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track y_var input: {e}")

@reactive.Effect
@reactive.event(input.dep_var)
def track_dep_var():
    """Track dep_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['dep_var'] = input.dep_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track dep_var input: {e}")

@reactive.Effect
@reactive.event(input.indep_var)
def track_indep_var():
    """Track indep_var input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['indep_var'] = input.indep_var()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track indep_var input: {e}")

@reactive.Effect
@reactive.event(input.covtype)
def track_covtype():
    """Track covtype input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['covtype'] = input.covtype()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track covtype input: {e}")

@reactive.Effect
@reactive.event(input.constant)
def track_constant():
    """Track constant input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['constant'] = input.constant()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track constant input: {e}")

@reactive.Effect
@reactive.event(input.column_search)
def track_column_search():
    """Track column_search input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['column_search'] = input.column_search()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track column_search input: {e}")

@reactive.Effect
@reactive.event(input.column_input)
def track_column_input():
    """Track column_input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['column_input'] = input.column_input()
        input_state_store.set(current_state)
    except Exception as e:
        logger.warning(f"Could not track column_input: {e}")

@reactive.Effect
@reactive.event(input.show_complete_cases)
def track_show_complete_cases():
    """Track show_complete_cases input value"""
    try:
        current_state = input_state_store.get() or {}
        current_state['show_complete_cases'] = input.show_complete_cases()
        input_state_store.set(current_state)
        
        # Clear filtered dataframe when checkbox state changes
        filtered_df.set(None)
    except Exception as e:
        logger.warning(f"Could not track show_complete_cases input: {e}")

@reactive.Effect
@reactive.event(input.load_dataset)
def clear_filtered_data_on_dataset_change():
    """Clear filtered dataframe when dataset changes"""
    try:
        filtered_df.set(None)
    except Exception as e:
        logger.warning(f"Could not clear filtered dataframe: {e}")