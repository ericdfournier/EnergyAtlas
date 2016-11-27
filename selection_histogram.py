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
    Spacer,
    Select,
    Range1d,
    PreText
    )
from bokeh.plotting import (
    figure, 
    curdoc,
    )

#%% create three normal population samples with different parameters

seed = (12345)
np.random.seed(seed)

size = 10000
mu = np.log([2000, 200])
sigma =[[0.2,0],[0,0.2]]

x_rnd,y_rnd = np.exp(np.random.multivariate_normal(mu, sigma, size).T)
x_rnd_y_rnd = np.divide(x_rnd,y_rnd)
year_rnd = np.random.randint(1850,2010,size)

df = pd.DataFrame()

df['size'] = x_rnd
df['consumption'] = y_rnd
df['intensity'] = x_rnd_y_rnd
df['year'] = year_rnd

#%% create stats

stats = PreText(text='', width=500)

#%% create input controls

axis_map = {
    "Energy Consumption [MBTU]": "consumption",
    "Building Size [sq.ft.]": "size",
    "Energy Consumption Intensity [MBTU / sq.ft.]" : "intensity",
    "Construction Vintage [Year]" : "year"
}

x_axis = Select(title='X-Axis', value='Building Size [sq.ft.]', options=sorted(axis_map.keys()))
y_axis = Select(title='Y-Axis', value='Energy Consumption [MBTU]', options=sorted(axis_map.keys()))

#%% create columnar data source

source = ColumnDataSource(data=dict(x=[],y=[]))

source.data = dict(
    x=df[axis_map[x_axis.value]],
    y=df[axis_map[y_axis.value]],
)

#%% create the scatter plot

TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

p = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=50, 
       min_border_left=50, toolbar_location="right", 
       x_axis_location="above", y_axis_location="left", 
       title="Linked Histograms", webgl=True)

p.background_fill_color = None
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False    

r = p.scatter('x', 'y', source=source, size=3, color="#3A5785", alpha=0.6, selection_color="orange")

# %% create the horizontal histogram

hhist, hedges = np.histogram(np.array(source.data['x']), bins=20)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

LINE_ARGS_1 = dict(color="orange", line_color=None)
LINE_ARGS_2 = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, plot_width=p.plot_width, 
            plot_height=200, x_range=p.x_range, y_range=(-hmax, hmax), 
            min_border=10, min_border_left=50, y_axis_location="right", 
            webgl=True)

ph.yaxis.major_label_orientation = np.pi/4
ph.xaxis.axis_label = x_axis.value

ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, 
        color="white", line_color="#3A5785")
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, 
              alpha=0.5, **LINE_ARGS_1)
hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, 
              alpha=0.1, **LINE_ARGS_2)

#%% create the vertical histogram

vhist, vedges = np.histogram(np.array(source.data['y']), bins=20)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, plot_width=200, 
            plot_height=p.plot_height, x_range=(-vmax, vmax), 
            y_range=p.y_range, min_border=10, y_axis_location="right", 
            webgl=True)

pv.xaxis.major_label_orientation = np.pi/4
pv.yaxis.axis_label = y_axis.value

pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, 
        color="white", line_color="#3A5785")
vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, 
              alpha=0.5, **LINE_ARGS_1)
vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, 
              alpha=0.1, **LINE_ARGS_2)

#%% Update Figure
    
def update_figure():
                
    source.data = dict(
        x=df[axis_map[x_axis.value]],
        y=df[axis_map[y_axis.value]],
    )
        
    ph.xaxis.axis_label = x_axis.value
    pv.yaxis.axis_label = y_axis.value    
    
    minx = min(source.data['x'])
    maxx = max(source.data['x'])
    miny = min(source.data['y'])
    maxy = max(source.data['y'])
    
    p.x_range = Range1d(minx-(0.1*maxx),maxx+(0.1*maxx))
    p.y_range = Range1d(miny-(0.1*maxy),maxy+(0.1*maxy))
    
    ph.x_range = p.x_range
    pv.y_range = p.y_range

#%% Update Data

def update_stats(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        stats.text = str(df.describe())
    else:
        stats.text = str(df.ix[inds].describe())
        
#%% Update Histograms

def update_minor_histograms(inds):
    
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        hhist1, hhist2 = hzeros, hzeros
        vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(np.array(source.data['x']), dtype=np.bool)
        neg_inds[inds] = False

        hhist1, _ = np.histogram(np.array(source.data['x'][inds]), bins=hedges)
        hhist2, _ = np.histogram(np.array(source.data['x'][neg_inds]), bins=hedges)
        vhist1, _ = np.histogram(np.array(source.data['y'][inds]), bins=vedges)
        vhist2, _ = np.histogram(np.array(source.data['y'][neg_inds]), bins=vedges)

    hh1.data_source.data["top"]   =  hhist1
    hh2.data_source.data["top"]   = -hhist2
    vh1.data_source.data["right"] =  vhist1    
    vh2.data_source.data["right"] = -vhist2

#%% Update Selection    
    
def update_selection(attr, old, new):
    
    update_figure()
        
    inds = np.array(new['1d']['indices'], dtype=int)
    
    update_stats(inds)
    update_minor_histograms(inds)

#%% Create Selection Widgets

controls = [x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update_figure())    
    
sizing_mode = 'fixed' 

update_figure()

widgets = widgetbox(*controls, sizing_mode=sizing_mode)
layout = row(column(widgets,stats), column(row(p, pv), row(ph, Spacer(width=200, height=200))))

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

r.data_source.on_change('selected', update_selection)

# TODO: Integrate slider functionality into selection update callback

#min_vintage.on_change('value', update_selection)
#max_vintage.on_change('value', update_selection)

