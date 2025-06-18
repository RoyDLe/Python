import seaborn as sns
import pandas as pd
from shared import df, filename
import plotly.express as px
from shinywidgets import render_widget  
from Regmodels import OLS
from shiny import reactive

from shiny.express import input, render, ui

model_store = reactive.Value()
ui.page_opts(title="Navigate")

ui.nav_spacer()

def get_numeric_columns():
    return list(df.select_dtypes(include=['float64', 'int64']).columns)

def get_columns():
    return list(df.columns)

with ui.nav_panel("Raw Data"):
    
    with ui.navset_card_underline(title=filename):
        with ui.nav_panel("Table"):
            ui.input_checkbox_group(
                "selected_columns",
                "Select columns to display:",
                choices=get_columns(),
                selected=get_columns(),
        )
            @render.data_frame
            def data():
                l = list(input.selected_columns())
                columns = list(l)
                return df[columns]
            
        with ui.nav_panel("Frequency Plot"):
            ui.input_select(
                "var", 
                "Select variable", 
                choices=get_numeric_columns()
            )
            ui.input_text(
                'bins',
                'Number of bins:',
                value = 50
            )
            ui.input_text(
                'bw_adjust',
                'Bandwidth adjustment:',
                value = 1.0
            )
            @render.plot
            def hist():
                try:
                    bins = int(input.bins())
                except ValueError:
                   return None
                
                try:
                    bw = float(input.bw_adjust())
                except ValueError:
                   return None

                p = sns.histplot(
                    df, 
                    x=input.var(), 
                    stat="density",
                    facecolor="#e6bc32", 
                    edgecolor="#ebd9b2",
                    alpha=0.8,
                    bins = bins,
                )
                sns.kdeplot(df[input.var()], 
                            color="#5040ff", 
                            linewidth=2,
                            bw_adjust=bw
                        )
                
                return p.set(xlabel=None)
            
        with ui.nav_panel('Correlations'):
             ui.input_checkbox_group(
                "selected_numerical",
                "Select columns to display:",
                choices=get_numeric_columns(),
                selected=get_numeric_columns(),
            )
             @render_widget
             def corr():
                if len(input.selected_numerical()) > 1:
                    p = px.imshow(df[list(input.selected_numerical())].corr(), 
                        title="Correlation Heatmap",
                        color_continuous_scale='YlOrRd', 
                        labels={'x': 'Features', 'y': 'Features'}, 
                        width=600, height=600)

                    p.update_layout(
                        title_font=dict(size = 20, family = 'Verdana'),
                        template = 'plotly_white',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    return p
                else:
                    return None

with ui.nav_panel("Summary Data"):
    "This is the second 'page'."

with ui.nav_panel("2D Regression Plots"):
    ui.input_select(
        "x_var",
        "Select X-axis variable",
        choices=get_numeric_columns()
    )
    ui.input_select(
        "y_var",
        "Select Y-axis variable",
        choices=get_numeric_columns()
    )

    @render_widget
    def regression():
        p = px.scatter(
            x=df[input.x_var()],
            y=df[input.y_var()],
            trendline='ols',
            trendline_color_override="#ab265f",
            title='Regression Plot'
        )
        p.update_traces(marker=dict(size=8, opacity=0.8, color="#e6bc32", line=dict(width=1, color="black")))
        p.update_layout(
            template="plotly_white", 
            plot_bgcolor="#f9f9f9",  
            xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False),
            font=dict(family="Arial", size=14),
            title_font=dict(size=20, family="Verdana"),
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_title = input.x_var(),
            yaxis_title = input.y_var()
        )
        p.add_annotation(
            x=df[input.x_var()].mean(), y=df[input.y_var()].mean(),
            text="Mean",
            showarrow=True, arrowhead=2, arrowcolor="black",
            font=dict(size=14, color="black")
        )
        return p

with ui.nav_panel("Multiple Regression"):
     with ui.navset_card_underline(title=filename):
        with ui.nav_panel("Settings"):
            ui.input_select(
                'dep_var',
                'Select dependent variable:',
                choices = get_numeric_columns()
            )
            @render.ui
            def dynamic_indep_vars():
                # Filter out the dependent variable from choices
                choices = [x for x in get_columns() if x != input.dep_var()]
                return ui.input_checkbox_group(
                    "indep_var",
                    "Select independent variables:",
                    choices=choices,
                    selected=choices,
                )
            ui.input_select(
                'covtype',
                'Select covariance type:',
                choices = ['standard', 'white', 'hc1', 'hc2']
            )
            ui.input_checkbox(
                'constant',
                'Include constant',
                value=True
            )
        with ui.nav_panel("Results"):
           @render.data_frame
           def results():
               if not input.indep_var():
                   return pd.DataFrame({'Message': ['Please select at least one independent variable']})
               
               model = OLS(
                   df[input.dep_var()],
                   df[list(input.indep_var())]
               )
               model_store.set(model)
               table = model.fit(covtype=input.covtype(), constant=input.constant())
               return table
           
        with ui.nav_panel("Diagnostics"):
            @render_widget
            def residual_plot():
                if model_store.get() is None:
                    return None
                    
                model = model_store.get()
                p = px.scatter(
                    x=model.fitted,
                    y=model.residuals,
                    labels={'x': 'Fitted values', 'y': 'Residuals'},
                    title='Residual Plot'
                )
                p.add_hline(y=0, line_dash="dash", line_color="red")
                p.update_traces(marker=dict(size=8, opacity=0.8, color="#e6bc32", line=dict(width=1, color="black")))
                p.update_layout(
                    template="plotly_white",
                    plot_bgcolor="#f9f9f9",
                    xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False),
                    font=dict(family="Arial", size=14),
                    title_font=dict(size=20, family="Verdana"),
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                return p