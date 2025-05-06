import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import networkx as nx
import os
from datetime import datetime

# Configure Plotly for Spyder
import plotly.io as pio
try:
    __IPYTHON__  # Will raise NameError if not in IPython
    pio.renderers.default = 'browser'  # Use browser for Spyder compatibility
except NameError:
    pio.renderers.default = 'browser'

# Initialize notebook mode
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

class HARVisualizer:
    # Professional color scheme
    COLOR_SCHEME = {
        'background': '#f8f9fa',
        'text': '#2c3e50',
        'primary': '#3498db',
        'secondary': '#e74c3c',
        'accent': '#2ecc71'
    }
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        self.prepare_visualizations()
        
    def load_data(self):
        """Load processed data with proper type conversion"""
        self.features = pd.read_parquet(os.path.join(self.data_dir, "window_features.parquet"))
        self.graph = nx.read_graphml(os.path.join(self.data_dir, "sensor_graph.graphml"))
        
        # Convert timestamp and ensure proper datetime types
        self.features['timestamp'] = pd.to_datetime(self.features['timestamp'])
        self.features['hour'] = self.features['timestamp'].dt.hour
        self.features['date'] = self.features['timestamp'].dt.date
        
        # Add default positions if missing
        for i, node in enumerate(self.graph.nodes()):
            if 'pos' not in self.graph.nodes[node]:
                self.graph.nodes[node]['pos'] = (i % 10, i // 10)
            if 'room' not in self.graph.nodes[node]:
                self.graph.nodes[node]['room'] = 'Unknown'

    def prepare_visualizations(self):
        """Prepare interactive dashboard components"""
        # Add CSS styling
        display(widgets.HTML('''
        <style>
            .widget-label { font-weight: bold !important; }
            .widget-slider .ui-slider-handle { background: #4CAF50 !important; }
            .widget-dropdown select { background-color: #f8f9fa !important; }
            .widget-readout { font-weight: bold !important; }
        </style>
        '''))
        
        # Convert dates to strings for widget compatibility
        date_options = sorted(self.features['date'].unique())
        date_str_options = [date.strftime('%Y-%m-%d') for date in date_options]
        
        self.date_picker = widgets.Dropdown(
            options=date_str_options,
            value=date_str_options[0] if date_str_options else None,
            description='📅 Select Date:',
            style={'description_width': 'initial'},
            layout={'width': '300px'}
        )
        
        self.activity_selector = widgets.SelectMultiple(
            options=sorted(self.features['dominant_activity'].unique()),
            description='🏃 Filter Activities:',
            style={'description_width': 'initial'},
            layout={'width': '300px'}
        )
        
        self.hour_slider = widgets.IntRangeSlider(
            value=[0, 24],
            min=0,
            max=24,
            step=1,
            description='🕒 Hour Range:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.sensor_threshold = widgets.IntSlider(
            value=5,
            min=0,
            max=50,
            step=1,
            description='📶 Sensor Threshold:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        # Set up interactive output
        self.output = widgets.Output()
        self.output.layout = {'border': '1px solid #ddd', 'padding': '10px'}
        
        # Create controls container
        controls = widgets.VBox([
            self.date_picker,
            self.activity_selector,
            self.hour_slider,
            self.sensor_threshold
        ], layout=widgets.Layout(border='1px solid #ddd', padding='10px'))
        
        # Dashboard title
        dashboard_title = widgets.HTML(
            value='<h1 style="text-align:center; color:#2c3e50;">Human Activity Recognition Dashboard</h1>'
        )
        
        # Create dashboard layout
        self.dashboard = widgets.VBox([
            dashboard_title,
            widgets.HBox([controls, self.output])
        ])
        
        # Set up observers
        self.date_picker.observe(lambda _: self.update_dashboard(), names='value')
        self.activity_selector.observe(lambda _: self.update_dashboard(), names='value')
        self.hour_slider.observe(lambda _: self.update_dashboard(), names='value')
        self.sensor_threshold.observe(lambda _: self.update_dashboard(), names='value')

    def create_dashboard(self):
        """Create the interactive dashboard layout"""
        display(self.dashboard)
        self.update_dashboard()

    def update_dashboard(self):
        """Update visualizations based on widget selections"""
        with self.output:
            self.output.clear_output()
            try:
                selected_date = self.date_picker.value
                selected_activities = self.activity_selector.value
                hour_range = self.hour_slider.value
                sensor_threshold = self.sensor_threshold.value
                
                if not selected_date:
                    print("Please select a date")
                    return
                    
                selected_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
                filtered = self.features[
                    (self.features['date'] == selected_date) &
                    (self.features['hour'] >= hour_range[0]) &
                    (self.features['hour'] <= hour_range[1])
                ].copy()
                
                if selected_activities:
                    filtered = filtered[filtered['dominant_activity'].isin(selected_activities)]
                
                if len(filtered) == 0:
                    print("No data matches the selected filters")
                    return
                
                # Activity Overview Section
                display(widgets.HTML(
                    '<h2 style="color:#2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;">Activity Overview</h2>'
                ))
                self.create_activity_summary(filtered)
                self.create_timeline(filtered)
                self.create_activity_donut(filtered)
                
                # Sensor Insights Section
                display(widgets.HTML(
                    '<h2 style="color:#2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;">Sensor Insights</h2>'
                ))
                self.create_sensor_heatmap(filtered, sensor_threshold)
                self.create_sensor_treemap(filtered)
                self.create_activity_graph(filtered)
                
                # Time Patterns Section
                display(widgets.HTML(
                    '<h2 style="color:#2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;">Time Patterns</h2>'
                ))
                self.create_hourly_patterns(filtered)
                
            except Exception as e:
                print(f"Error updating dashboard: {str(e)}")

    def create_timeline(self, data):
        """Create activity timeline visualization"""
        try:
            fig = px.timeline(
                data,
                x_start="timestamp",
                x_end=data['timestamp'] + pd.Timedelta(minutes=5),
                y="dominant_activity",
                color="dominant_activity",
                title="<b>Activity Timeline</b>",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(
                height=500,
                font=dict(family="Arial", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Time",
                yaxis_title="Activity",
                legend_title="Activity Types",
                hovermode="x unified",
                margin=dict(t=40, b=20, l=20, r=20)
            )
            
            fig.update_xaxes(
                tickformat="%H:%M",
                rangeslider_visible=True
            )
            
            fig.show(renderer=pio.renderers.default)
        except Exception as e:
            print(f"Error creating timeline: {str(e)}")

    def create_sensor_heatmap(self, data, threshold):
        """Create sensor activation heatmap"""
        try:
            sensor_cols = [col for col in data.columns if col.startswith(('M', 'D'))]
            sensor_activations = data[sensor_cols].sum().sort_values(ascending=False)
            sensor_activations = sensor_activations[sensor_activations > threshold]
            
            if len(sensor_activations) > 0:
                fig = px.bar(
                    sensor_activations,
                    orientation='h',
                    labels={'value': 'Activation Count', 'index': 'Sensor ID'},
                    title=f"<b>Sensor Activations (Threshold: {threshold})</b>",
                    color=sensor_activations.values,
                    color_continuous_scale='Viridis',
                    template="plotly_white"
                )
                
                fig.update_layout(
                    height=600,
                    font=dict(family="Arial", size=12),
                    xaxis_title="Number of Activations",
                    yaxis_title="Sensor ID",
                    coloraxis_showscale=False,
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                
                fig.show(renderer=pio.renderers.default)
            else:
                print(f"No sensors exceeded activation threshold of {threshold}")
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")

    def create_activity_graph(self, data):
        """Create interactive sensor network graph"""
        try:
            sensor_cols = [col for col in data.columns if col.startswith(('M', 'D'))]
            active_sensors = data[sensor_cols].sum()
            active_sensors = active_sensors[active_sensors > 0].index.tolist()
            
            if not active_sensors:
                print("No active sensors found for the selected filters")
                return
                
            # Create a subgraph for only active sensors
            subgraph = self.graph.subgraph(active_sensors)
            pos = nx.get_node_attributes(subgraph, 'pos')
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            for node in subgraph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                activations = data[node].sum()
                node_text.append(
                    f"<b>Sensor {node}</b><br>"
                    f"Room: {subgraph.nodes[node]['room']}<br>"
                    f"Activations: {activations}<br>"
                    f"Connections: {subgraph.degree[node]}"
                )
                node_size.append(10 + activations/2)
                node_color.append(activations)
                
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[node.split('_')[-1] for node in subgraph.nodes()],
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='YlOrRd',
                    size=node_size,
                    color=node_color,
                    colorbar=dict(
                        thickness=15,
                        title='Activations',
                        xanchor='left'
                    ),
                    line_width=1
                ),
                textfont=dict(size=10),
                hovertext=node_text,
                hoverinfo='text'
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
    layout=go.Layout(
        title=dict(text='<b>Sensor Network Activity</b>', font=dict(size=16)),
                             showlegend=False,
                             hovermode='closest',
                             margin=dict(b=20,l=5,r=5,t=40),
                             height=700,
                             template="plotly_white",
                             xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(showgrid=False, zeroline=False)
                         ))
            
            fig.show(renderer=pio.renderers.default)
        except Exception as e:
            print(f"Error creating activity graph: {str(e)}")

    def create_activity_summary(self, data):
        """Create summary cards for key metrics"""
        try:
            total_activities = len(data)
            unique_activities = data['dominant_activity'].nunique()
            avg_duration = "5 min"  # Fixed based on your window size
            
            cards = widgets.HBox([
                self._create_metric_card("📊 Total Activities", total_activities, "#3498db"),
                self._create_metric_card("🔄 Activity Types", unique_activities, "#e74c3c"),
                self._create_metric_card("⏱ Avg Duration", avg_duration, "#2ecc71")
            ])
            
            display(cards)
        except Exception as e:
            print(f"Error creating activity summary: {str(e)}")
    
    def _create_metric_card(self, title, value, color):
        """Helper to create metric cards"""
        return widgets.VBox([
            widgets.HTML(f"<h3 style='color:{color}; margin-bottom:0;'>{title}</h3>"),
            widgets.HTML(f"<h1 style='color:{color}; margin-top:0;'>{value}</h1>")
        ], layout=widgets.Layout(
            border=f'2px solid {color}',
            padding='10px',
            margin='5px',
            width='200px'
        ))

    def create_activity_donut(self, data):
        """Create donut chart of activity distribution"""
        try:
            activity_counts = data['dominant_activity'].value_counts().reset_index()
            activity_counts.columns = ['Activity', 'Count']
            
            fig = px.pie(
                activity_counts,
                names='Activity',
                values='Count',
                hole=0.4,
                title="<b>Activity Distribution</b>",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}"
            )
            
            fig.show(renderer=pio.renderers.default)
        except Exception as e:
            print(f"Error creating activity donut: {str(e)}")

    def create_sensor_treemap(self, data):
        """Create treemap of sensor activations by room"""
        try:
            sensor_cols = [col for col in data.columns if col.startswith(('M', 'D'))]
            sensor_activations = data[sensor_cols].sum().reset_index()
            sensor_activations.columns = ['sensor', 'activations']
            
            # Add room information
            sensor_activations['room'] = sensor_activations['sensor'].apply(
                lambda x: self.graph.nodes[x]['room'] if x in self.graph.nodes else 'Unknown'
            )
            
            fig = px.treemap(
                sensor_activations,
                path=['room', 'sensor'],
                values='activations',
                title="<b>Sensor Activations by Room</b>",
                color='activations',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                height=500,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            
            fig.show(renderer=pio.renderers.default)
        except Exception as e:
            print(f"Error creating sensor treemap: {str(e)}")

    def create_hourly_patterns(self, data):
        """Create line chart of activity patterns by hour"""
        try:
            hourly_data = data.groupby(['hour', 'dominant_activity']).size().unstack().fillna(0)
            
            fig = px.line(
                hourly_data,
                x=hourly_data.index,
                y=hourly_data.columns,
                title="<b>Activity Patterns by Hour</b>",
                labels={'value': 'Activity Count', 'hour': 'Hour of Day'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(
                height=400,
                xaxis=dict(
                    tickmode='linear',
                    dtick=1
                ),
                yaxis=dict(
                    rangemode='tozero'
                ),
                hovermode='x unified',
                legend=dict(
                    title='Activities',
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            
            fig.show(renderer=pio.renderers.default)
        except Exception as e:
            print(f"Error creating hourly patterns: {str(e)}")

if __name__ == "__main__":
    visualizer = HARVisualizer(r"C:\Users\User\Downloads\aruba\processed")
    visualizer.create_dashboard()