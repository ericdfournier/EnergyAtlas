#%% package imports

import numpy as np
import pandas as pd

from bokeh.layouts import (
    row, 
    column,
    widgetbox
    )
from bokeh.models import (
    ColumnDataSource,
    BoxSelectTool, 
    LassoSelectTool,
    Select,
    Range1d,
    PreText
    )
from bokeh.plotting import (
    figure, 
    curdoc,
    )

#%% Generate Random Energy Data for Visualization

seed = (12345)
np.random.seed(seed)

size = 10000
mu = np.log([2000, 200])
sigma =[[0.2,0],[0,0.2]]

x_rnd,y_rnd = np.exp(np.random.multivariate_normal(mu, sigma, size).T)
x_rnd_y_rnd = np.divide(x_rnd,y_rnd)
year_rnd = np.random.randint(1850,2010,size)

#%% create reference data frame

df = pd.DataFrame()

df['size'] = x_rnd
df['consumption'] = y_rnd
df['intensity'] = x_rnd_y_rnd
df['year'] = year_rnd

#%% create axis map

axis_map = {
    "Energy Consumption [MBTU]": "consumption",
    "Building Size [sq.ft.]": "size",
    "Energy Consumption Intensity [MBTU / sq.ft.]" : "intensity",
    "Construction Vintage [Year]" : "year"
}

#%% create plot columnar data source

plot_source = ColumnDataSource(data=dict(
    x = [],
    y = []
))

#%% create stats

stats = PreText(text='', width=500)

#%% create the scatter plot

def create_scatter(plot_source):

    PLOT_TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"
    
    p = figure(tools=PLOT_TOOLS, plot_width=700, plot_height=700, 
               min_border=50, min_border_left=50, toolbar_location="right", 
               x_axis_location="above", y_axis_location="left", 
               title="Linked Histograms", webgl=True)
    
    p.background_fill_color = None
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False  
    
    p.circle('x', 'y', source=plot_source, size=3, color="#3A5785", alpha=0.6, 
             selection_color="orange", nonselection_alpha=0.1, 
             selection_alpha=0.4)

    return p
    
    # %% create the horizontal histogram

def create_major_x_histogram(plot_source, inds):
    
    x = plot_source.data['x']

    if len(inds) == 0 or len(inds) == len(plot_source.data['x']):
        hhist0, hedges0 = np.histogram(np.array(x), bins=20)
    else:
        hhist0, hedges0 = np.histogram(np.array(x[inds]), bins=20)
        
    hmax0 = max(hhist0)*1.1

    maxx = max(plot_source.data['x'])*1.1
    minx = min(plot_source.data['x'])-(max(plot_source.data['x'])*0.1) 
    
    ph = figure(toolbar_location=None, plot_width=p.plot_width, 
                plot_height=200, x_range= Range1d(end=maxx, start=minx), 
                y_range = Range1d(end=hmax0,start=-hmax0),
                min_border=10, min_border_left=50, y_axis_location="right", 
                webgl=True)
    
    ph.yaxis.major_label_orientation = np.pi/4
    ph.xaxis.axis_label = x_axis.value
    
    hh0 = ph.quad(bottom=0, left=hedges0[:-1], right=hedges0[1:], top=hhist0, 
            color="white", line_color="#3A5785")

    return ph, hh0
    
#%% Update X-axis
    
def update_x_axis(attr, old, new):
    
    p.xaxis.axis_label = x_axis.value   
                
    plot_source.data['x'] = df[axis_map[new]].values    
    maxx = max(plot_source.data['x'])*1.1
    minx = min(plot_source.data['x'])-(max(plot_source.data['x'])*0.1) 
    
    x = plot_source.data['x']

    if len(inds) == 0 or len(inds) == len(plot_source.data['x']):
        hhist0, hedges0 = np.histogram(np.array(x), bins=20)
    else:
        hhist0, hedges0 = np.histogram(np.array(x[inds]), bins=20)
        
    hmax0 = max(hhist0)*1.1

    ph.x_range = Range1d(end=maxx, start=minx)
    ph.y_range = Range1d(end=hmax0,start=-hmax0)
    
    hh0.data_source.data['left'] = hedges0[:-1]
    hh0.data_source.data['right'] = hedges0[1:]
    hh0.data_source.data['top'] = hhist0
    
    
#%% Update Y-axis

def update_y_axis(attr, old, new):
    
    plot_source.data['y'] = df[axis_map[new]].values

    maxy = max(plot_source.data['y'])*1.1
    miny = min(plot_source.data['y'])-(max(plot_source.data['y'])*0.1)
    
    p.y_range = Range1d(end=maxy,start=miny)
    p.yaxis.axis_label = y_axis.value
    
    #%% Update Stats Panel

def update_stats(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        stats.text = str(df.describe())
    else:
        stats.text = str(df.ix[inds].describe())
    
#%% Update Selection    
    
def update_selection(attr, old, new):
            
    inds = np.array(new['1d']['indices'], dtype=int)
    update_stats(inds)
    
    return inds
    
#%% Update plot source data

def update_data():
    
    plot_source.data['x'] = df[axis_map[x_axis.value]].values
    plot_source.data['y'] = df[axis_map[y_axis.value]].values
        
#%% create axes selectors

x_axis = Select(title='X-Axis', value='Building Size [sq.ft.]', 
                options=sorted(axis_map.keys()))
y_axis = Select(title='Y-Axis', value='Energy Consumption [MBTU]', 
                options=sorted(axis_map.keys()))

#%% Generate Plot Variables

inds = np.asarray([], dtype=int)
update_data()
p = create_scatter(plot_source)
ph, hh0 = create_major_x_histogram(plot_source, inds)

#%% Create Widgets and Layout

sizing_mode = 'fixed'
widgets = widgetbox([x_axis, y_axis], sizing_mode=sizing_mode)
layout = row(column(widgets, stats), column(row(p), row(ph)))
curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

x_axis.on_change('value', update_x_axis)
y_axis.on_change('value', update_y_axis)
plot_source.on_change('selected', update_selection)