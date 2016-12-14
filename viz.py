#%% Package Imports

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
    HoverTool,
    Select,
    PreText
    )
from bokeh.plotting import (
    figure, 
    curdoc,
    )

#%% Load Static Map Datasource

path = "/Users/edf/Repositories/EnergyAtlas/data/la_county_neighborhood_boundaries_single_simple.json"
map_source = pd.read_json(path, typ='series', orient='column')
neighborhoods = {}

#%% Clean Raw Static Map Datasource

for i,feature in enumerate(map_source.features):
        
    if feature['geometry']['type'] == 'Polygon':
        
        coords = feature['geometry']['coordinates'][0]
        name = feature['properties']['Name'][:]
        lon,lat,_ = np.reshape(coords,[len(coords),3]).T
        neighborhoods[i] = {}
        neighborhoods[i]['lat'] = lat
        neighborhoods[i]['lon'] = lon
        neighborhoods[i]['name'] = name
        
    elif feature['geometry']['type'] == 'MultiPolygon':
        
        coords = feature['geometry']['coordinates'][:]
        name = feature['properties']['Name'][:]
        lons = []
        lats = []
        
        for j, poly in enumerate(coords):
            
            lon,lat,_ = np.reshape(coords[j][0],[len(coords[j][0]),3]).T
            lons.append(lon)
            lats.append(lat)
        
        neighborhoods[i] = {}
        neighborhoods[i]['lat'] = lats
        neighborhoods[i]['lon'] = lons
        neighborhoods[i]['name'] = name

#%% Prep Static Map Source Data for Plotting

neighborhood_lon = np.asarray([neighborhood["lon"] for neighborhood in neighborhoods.values()])
neighborhood_lat = np.asarray([neighborhood["lat"] for neighborhood in neighborhoods.values()])
neighborhood_names = np.asarray([neighborhood['name'] for neighborhood in neighborhoods.values()])
neighborhood_avg = np.random.randint(100,300,len(neighborhood_names))       
color = np.asarray(["white"]*len(neighborhood_names))

#%% Create Map columnar data source

map_source = ColumnDataSource(data=dict(
    lon = neighborhood_lon,
    lat = neighborhood_lat,
    avg_consumption = neighborhood_avg,
    name = neighborhood_names,
    color = color
))             

#%% Generate Random Energy Data for Visualization

seed = (12345)
np.random.seed(seed)

size = 1000
mu = np.log([2000, 200])
sigma =[[0.2,0],[0,0.2]]

x_rnd,y_rnd = np.exp(np.random.multivariate_normal(mu, sigma, size).T)
x_rnd_y_rnd = np.divide(x_rnd,y_rnd)
year_rnd = np.random.randint(1850,2010,size)

unique_names = np.asarray(list(set(neighborhood_names)))
neighborhoods = [None]*size

for i in range(size):
    ind = np.random.randint(0,len(unique_names),1,dtype=int)
    neighborhoods[i] = unique_names[ind]

neighborhoods = np.asarray(neighborhoods)

#%% Create Reference Data Frame

df = pd.DataFrame()

df['size'] = x_rnd
df['consumption'] = y_rnd
df['intensity'] = x_rnd_y_rnd
df['year'] = year_rnd
df['name'] = neighborhoods

#%% Create Axis Map

axis_map = {
    "Energy Consumption [MBTU]": "consumption",
    "Building Size [sq.ft.]": "size",
    "Energy Consumption Intensity [MBTU / sq.ft.]" : "intensity",
    "Construction Vintage [Year]" : "year"
}

#%% Create Plot Columnar Data Source

plot_source = ColumnDataSource(data=dict(
    x = [],
    y = []
))

#%% Create Stats Widget

stats = PreText(text='', width=500)

#%% Create Map Plot

def create_map(map_source):
    
    MAP_TOOLS="pan,wheel_zoom,reset,hover,save"
    
    m = figure(title="Los Angeles County Neighborhoods", tools=MAP_TOOLS, 
               plot_width=700, plot_height=900, min_border=10, 
               x_axis_location="below", y_axis_location="left",webgl=True,
               toolbar_location="above")
    m.yaxis.axis_label = "Latitude [DD]"
    m.xaxis.axis_label = "Longitude [DD]"
    
    m.patches('lon','lat', source=map_source,
              fill_color='color',
              line_color='#3A5785', line_width=0.5)
              
    hover = m.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("Name", "@name"),
        ("(Long, Lat)", "($x, $y)"),
        ("Mean Annual Energy Consumption [MBTU / Megaparcel]", "@avg_consumption")
    ]

    return m

#%% Create Scatter Plot

def create_scatter(plot_source):

    PLOT_TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"
    
    p = figure(tools=PLOT_TOOLS, plot_width=900, plot_height=900, 
               min_border=50, min_border_left=50, toolbar_location="above", 
               x_axis_location="below", y_axis_location="left", 
               title="Pairwise Scatterplot", webgl=True)
    
    p.yaxis.axis_label = y_axis.value
    p.xaxis.axis_label = x_axis.value
    
    p.background_fill_color = None
    p.select(BoxSelectTool).select_every_mousemove = False
    p.select(LassoSelectTool).select_every_mousemove = False  
    
    p.circle('x', 'y', source=plot_source, size=3, color="#3A5785", alpha=1.0, 
             selection_color="orange", nonselection_alpha=1.0, 
             selection_alpha=1.0)

    return p
    
#%% Create Y axis histogram

def create_x_histogram(plot_source):
        
    xhist0, xedges0 = np.histogram(np.array(plot_source.data['x']), bins=20)
    px = figure(toolbar_location=None, plot_width=500, 
                plot_height=300, min_border=10, title="X-Axis Histogram",
                y_axis_location="left", x_axis_location="below",
                webgl=True)
    px.xaxis.axis_label = x_axis.value
    px.yaxis.axis_label = 'Counts('+ x_axis.value + ')'
    xh = px.quad(bottom=0, left=xedges0[:-1], right=xedges0[1:], top=xhist0, 
            color="white", line_color="#3A5785")
    
    return px, xh
    
#%% Create Y axis histogram

def create_y_histogram(plot_source):
    
    yhist0, yedges0 = np.histogram(np.array(plot_source.data['y']), bins=20)
    py = figure(toolbar_location=None, plot_width=500, 
                plot_height=300, min_border=10, title="Y-Axis Histogram",
                y_axis_location="left", x_axis_location="below",
                webgl=True)
    py.xaxis.axis_label = y_axis.value
    py.yaxis.axis_label = 'Counts('+ y_axis.value + ')'
    yh = py.quad(bottom=0, left=yedges0[:-1], right=yedges0[1:], top=yhist0, 
            color="white", line_color="#3A5785")
    
    return py, yh
    
#%% Update X axis
    
def update_x_axis(attr, old, new):
    
    p.xaxis.axis_label = x_axis.value               
    plot_source.data['x'] = df[axis_map[new]].values
    xhist0, xedges0 = np.histogram(np.array(plot_source.data['x']), bins=20)  
    xh.data_source.data['right'] = xedges0[1:]
    xh.data_source.data['left'] = xedges0[:-1]
    xh.data_source.data['top'] = xhist0  

#%% Update Y axis

def update_y_axis(attr, old, new):
    
    p.yaxis.axis_label = y_axis.value
    plot_source.data['y'] = df[axis_map[new]].values
    yhist0, yedges0 = np.histogram(np.array(plot_source.data['y']), bins=20)  
    yh.data_source.data['right'] = yedges0[1:]
    yh.data_source.data['left'] = yedges0[:-1]
    yh.data_source.data['top'] = yhist0  
    
#%% Update Stats Panel Widget

def update_stats(inds):
        
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        stats.text = str(df.describe())
    else:
        stats.text = str(df.ix[inds].describe())
        
#%% Update Map

def update_map(inds):
    
    if len(inds) == 0 or len(inds) == len(df['size'].values):
        map_source.data['lon'] = neighborhood_lon
        map_source.data['lat'] = neighborhood_lat
        map_source.data['avg_consumption'] = neighborhood_avg
        map_source.data['name'] = neighborhood_names
        map_source.data['color'] = np.asarray(["white"]*len(neighborhood_names), dtype=object)
    else:
        color = np.asarray(["white"]*len(neighborhood_names), dtype=object)
        cur_names = np.unique(neighborhoods[inds])
        map_inds = np.in1d(neighborhood_names, cur_names)
        color[map_inds] = 'orange'
        map_source.data['color'] = color
    
#%% Update Selection    
    
def update_selection(attr, old, new):
            
    inds = np.array(new['1d']['indices'], dtype=int)
    update_stats(inds)
    update_map(inds)
    
    return inds
    
#%% Update Plot Source Data

def update_data():
    
    plot_source.data['x'] = df[axis_map[x_axis.value]].values
    plot_source.data['y'] = df[axis_map[y_axis.value]].values
        
#%% Create Axes Selector Widgets

x_axis = Select(title='X-Axis', value='Building Size [sq.ft.]', 
                options=sorted(axis_map.keys()))
y_axis = Select(title='Y-Axis', value='Energy Consumption [MBTU]', 
                options=sorted(axis_map.keys()))

#%% Generate Plot Variables

inds = np.asarray([], dtype=int)
update_data()
p = create_scatter(plot_source)
px, xh = create_x_histogram(plot_source)
py, yh = create_y_histogram(plot_source)
m = create_map(map_source)

#%% Generate Layout

sizing_mode = 'fixed'
widgets = widgetbox([x_axis, y_axis], sizing_mode=sizing_mode)
layout = row(column(widgets, stats, px, py), column(p), column(m))
curdoc().add_root(layout)
curdoc().title = "Selection Histogram"

x_axis.on_change('value', update_x_axis)
y_axis.on_change('value', update_y_axis)
plot_source.on_change('selected', update_selection)