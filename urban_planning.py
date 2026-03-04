import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, HeatMap, MarkerCluster
from folium.raster_layers import ImageOverlay, WmsTileLayer
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import io
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Set page configuration
st.set_page_config(layout="wide", page_title="SmartTown: Urban Planning Tool - Pakistan")

# Securely store API keys
OPENTOPOGRAPHY_API_KEY = os.getenv("OPENTOPOGRAPHY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define all available parameters
ALL_PARAMETERS = {
    "T2M": "🌡️ Average Temperature (°C) - Crucial for thermal comfort planning",
    "T2M_MAX": "🔥 Maximum Temperature (°C) - Informs cooling system requirements",
    "T2M_MIN": "❄️ Minimum Temperature (°C) - Guides heating infrastructure planning",
    "PRECTOTCORR": "🌧️ Daily Precipitation (mm) - Essential for stormwater management and flood prevention",
    "RH2M": "💧 Relative Humidity (%) - Impacts building material selection and HVAC planning",
    "ALLSKY_SFC_SW_DWN": "☀️ Solar Radiation (W/m^2) - Crucial for solar energy potential and natural lighting design",
    "WS2M": "💨 Wind Speed (m/s) - Influences building orientation and natural ventilation strategies",
    "T2MDEW": "💦 Dew Point (°C) - Important for moisture control in building design",
    "ALLSKY_SFC_LW_DWN": "🌞 Longwave Radiation (W/m^2) - Affects urban heat island mitigation strategies",
    "CLOUD_AMT": "☁️ Cloud Cover (%) - Impacts solar energy efficiency and natural lighting design",
    "GWETROOT": "🌱 Soil Moisture (%) - Crucial for green infrastructure and urban landscaping",
    "QV2M": "💨 Specific Humidity (kg/kg) - Informs HVAC system design for moisture control",
    "PS": "🏔️ Atmospheric Pressure (kPa) - Relevant for high-altitude urban planning",
    "T2MWET": "💧 Wet Bulb Temperature (°C) - Critical for assessing heat stress in urban areas",
    "ALLSKY_SFC_PAR_TOT": "🌿 Photosynthetically Active Radiation (W/m^2) - Important for urban agriculture and green space planning",
    "TOA_SW_DWN": "🛰️ Top-of-Atmosphere Solar Radiation (W/m^2) - Useful for advanced solar energy planning",
    "ALLSKY_SFC_SW_DNI": "🔆 Direct Normal Irradiance (W/m^2) - Critical for solar panel placement and efficiency",
    "ALLSKY_SRF_ALB": "🔦 Surface Albedo - Guides urban heat island mitigation and energy-efficient building design",
    "ALLSKY_SFC_SW_DIFF": "🌤️ Diffuse Solar Radiation (W/m^2) - Important for daylighting strategies in building design",
    "ALLSKY_KT": "📊 Solar Clearness Index - Helps in assessing overall solar energy potential for the area"
}

@st.cache_data(ttl=3600)
def get_nasa_power_data(lat, lon, start_date, end_date, selected_params):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(selected_params),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['properties']['parameter'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.replace(-999, np.nan)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch NASA data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_opentopography_data(south, north, west, east):
    dataset = 'SRTMGL1'
    output_format = 'GTiff'
    url = f'https://portal.opentopography.org/API/globaldem?demtype={dataset}&south={south}&north={north}&west={west}&east={east}&outputFormat={output_format}&API_Key={OPENTOPOGRAPHY_API_KEY}'
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                elevation_data = dataset.read(1)
                return elevation_data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch OpenTopography data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to process OpenTopography data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_osm_urban_data(south, north, west, east):
    """Fetch urban features from OpenStreetMap via Overpass API."""
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Two-part query: buildings (centers only) + other features (with geometry)
    query = f"""[out:json][timeout:90];
(way["building"]({south},{west},{north},{east}););
out center qt 5000;
(
  node["amenity"]({south},{west},{north},{east});
  way["highway"~"^(primary|secondary|tertiary|trunk|motorway|residential)$"]({south},{west},{north},{east});
  way["natural"="water"]({south},{west},{north},{east});
  way["waterway"~"^(river|stream|canal)$"]({south},{west},{north},{east});
  way["leisure"="park"]({south},{west},{north},{east});
  way["landuse"~"^(forest|grass|meadow)$"]({south},{west},{north},{east});
);
out body geom qt 5000;"""

    result = {'buildings': [], 'amenities': [], 'roads': [], 'water': [], 'green_spaces': []}

    try:
        response = requests.post(overpass_url, data={"data": query}, timeout=120)
        response.raise_for_status()
        data = response.json()

        for el in data.get('elements', []):
            tags = el.get('tags', {})

            if 'building' in tags and 'center' in el:
                result['buildings'].append([el['center']['lat'], el['center']['lon']])
            elif el['type'] == 'node' and 'amenity' in tags:
                result['amenities'].append({
                    'lat': el['lat'], 'lon': el['lon'],
                    'type': tags.get('amenity', ''),
                    'name': tags.get('name', '')
                })
            elif 'highway' in tags and 'geometry' in el:
                coords = [[pt['lat'], pt['lon']] for pt in el['geometry']]
                result['roads'].append({
                    'coords': coords,
                    'type': tags.get('highway', ''),
                    'name': tags.get('name', '')
                })
            elif ('waterway' in tags or tags.get('natural') == 'water') and 'geometry' in el:
                coords = [[pt['lat'], pt['lon']] for pt in el['geometry']]
                result['water'].append({
                    'coords': coords,
                    'type': tags.get('waterway', tags.get('natural', 'water')),
                    'name': tags.get('name', '')
                })
            elif ('leisure' in tags or 'landuse' in tags) and 'geometry' in el:
                if tags.get('leisure') == 'park' or tags.get('landuse') in ['forest', 'grass', 'meadow']:
                    coords = [[pt['lat'], pt['lon']] for pt in el['geometry']]
                    result['green_spaces'].append({
                        'coords': coords,
                        'type': tags.get('leisure', tags.get('landuse', '')),
                        'name': tags.get('name', '')
                    })
    except Exception as e:
        st.warning(f"Could not fetch OpenStreetMap data: {str(e)}")

    return result

def calculate_centroid(coordinates):
    lat_sum = sum(coord[1] for coord in coordinates[0])
    lon_sum = sum(coord[0] for coord in coordinates[0])
    count = len(coordinates[0])
    return lat_sum / count, lon_sum / count

def create_climate_visualizations(nasa_data):
    st.subheader("Climate Data Visualizations")

    plot_data = {
        "Temperature": ["T2M", "T2M_MAX", "T2M_MIN"],
        "Precipitation": ["PRECTOTCORR"],
        "Relative Humidity": ["RH2M"],
        "Solar Radiation": ["ALLSKY_SFC_SW_DWN"]
    }

    available_plots = [key for key, params in plot_data.items() if any(param in nasa_data.columns for param in params)]

    if not available_plots:
        st.warning("No data available for visualization. Please select more parameters.")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=available_plots[:4])

    for i, plot_type in enumerate(available_plots[:4], 1):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        for param in plot_data[plot_type]:
            if param in nasa_data.columns:
                if plot_type == "Precipitation":
                    fig.add_trace(go.Bar(x=nasa_data.index, y=nasa_data[param], name=ALL_PARAMETERS[param]), row=row, col=col)
                else:
                    fig.add_trace(go.Scatter(x=nasa_data.index, y=nasa_data[param], name=ALL_PARAMETERS[param]), row=row, col=col)

    fig.update_layout(height=800, width=1000, title_text=f"Climate Data ({nasa_data.index.min().date()} to {nasa_data.index.max().date()})")
    fig.update_xaxes(title_text="Date")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Additional Climate Parameters")
    for param in nasa_data.columns:
        if param not in sum(plot_data.values(), []):
            st.subheader(ALL_PARAMETERS[param])
            st.line_chart(nasa_data[param])

    st.download_button(
        label="Download Climate Data as CSV",
        data=nasa_data.to_csv(),
        file_name="climate_data.csv",
        mime="text/csv"
    )

    st.subheader("Climate Parameter Correlation Heatmap")
    corr = nasa_data.corr(numeric_only=True)
    fig_heatmap = go.Figure(data=go.Heatmap(
                   z=corr.values,
                   x=corr.columns,
                   y=corr.columns,
                   colorscale='Viridis'))
    fig_heatmap.update_layout(height=600, width=800, title_text="Parameter Correlation Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

def create_monthly_breakdown(nasa_data):
    st.subheader("Monthly Climate Summary")
    monthly = nasa_data.groupby(nasa_data.index.month).mean(numeric_only=True)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly.index = [month_names[i-1] for i in monthly.index]

    key_params = {
        'T2M': 'Avg Temperature (°C)',
        'PRECTOTCORR': 'Avg Precipitation (mm/day)',
        'ALLSKY_SFC_SW_DWN': 'Avg Solar Radiation (W/m²)',
        'RH2M': 'Avg Humidity (%)',
        'WS2M': 'Avg Wind Speed (m/s)'
    }
    available = {k: v for k, v in key_params.items() if k in monthly.columns}

    if not available:
        st.info("No key climate parameters selected for monthly breakdown.")
        return

    fig = make_subplots(
        rows=len(available), cols=1,
        subplot_titles=list(available.values()),
        shared_xaxes=True
    )
    for i, (param, label) in enumerate(available.items(), 1):
        if param == 'PRECTOTCORR':
            fig.add_trace(go.Bar(x=monthly.index, y=monthly[param], name=label,
                                 marker_color='steelblue'), row=i, col=1)
        else:
            fig.add_trace(go.Scatter(x=monthly.index, y=monthly[param], name=label,
                                     mode='lines+markers'), row=i, col=1)

    fig.update_layout(height=250 * len(available), showlegend=False,
                      title_text="Monthly Average Climate Parameters")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(monthly[list(available.keys())].rename(columns=available).round(2))

def create_wind_rose(nasa_data):
    if 'WS2M' not in nasa_data.columns:
        return
    st.subheader("Wind Speed Distribution")

    wind = nasa_data['WS2M'].dropna()
    bins = [0, 1, 2, 4, 6, 8, 100]
    labels = ['Calm (0-1)', 'Light (1-2)', 'Moderate (2-4)',
              'Fresh (4-6)', 'Strong (6-8)', 'Very Strong (8+)']
    wind_cat = pd.cut(wind, bins=bins, labels=labels)
    counts = wind_cat.value_counts().reindex(labels)

    fig = go.Figure(data=[go.Bar(
        x=labels, y=counts.values,
        marker_color=['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    )])
    fig.update_layout(
        title='Wind Speed Distribution',
        xaxis_title='Wind Category (m/s)',
        yaxis_title='Number of Days',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    if 'T2M' in nasa_data.columns:
        monthly_wind = nasa_data.groupby(nasa_data.index.month)['WS2M'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig2 = go.Figure(data=[go.Scatterpolar(
            r=[monthly_wind.get(i, 0) for i in range(1, 13)],
            theta=month_names,
            fill='toself',
            name='Wind Speed'
        )])
        fig2.update_layout(
            title='Monthly Wind Speed Pattern',
            polar=dict(radialaxis=dict(title='m/s')),
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

def calculate_flood_risk(nasa_data, elevation_data):
    st.subheader("Flood Risk Assessment")

    if 'PRECTOTCORR' not in nasa_data.columns:
        st.info("Precipitation data (PRECTOTCORR) not selected. Cannot compute flood risk.")
        return

    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))

    avg_precip = nasa_data['PRECTOTCORR'].mean()
    max_precip = nasa_data['PRECTOTCORR'].max()
    heavy_rain_days = (nasa_data['PRECTOTCORR'] > 10).sum()

    precip_risk = np.clip(avg_precip / 10, 0, 1)
    flat_area_pct = np.sum(slope < 2) / slope.size
    low_elevation_pct = np.sum(elevation_data < np.percentile(elevation_data, 25)) / elevation_data.size
    flood_risk_score = np.clip(
        0.4 * precip_risk + 0.35 * flat_area_pct + 0.25 * low_elevation_pct, 0, 1
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Daily Precipitation", f"{avg_precip:.2f} mm")
    col2.metric("Max Daily Precipitation", f"{max_precip:.2f} mm")
    col3.metric("Heavy Rain Days (>10mm)", f"{heavy_rain_days}")
    col4.metric("Flood Risk Score", f"{flood_risk_score:.2f}/1.00")

    if flood_risk_score > 0.7:
        st.error("HIGH flood risk. Significant drainage infrastructure and flood barriers recommended.")
    elif flood_risk_score > 0.4:
        st.warning("MODERATE flood risk. Stormwater management systems recommended.")
    else:
        st.success("LOW flood risk. Standard drainage should suffice.")

    slope_risk = np.clip(1 - slope / 30, 0, 1)
    elev_risk = np.clip(1 - (elevation_data - elevation_data.min()) /
                        (elevation_data.max() - elevation_data.min() + 1e-6), 0, 1)
    spatial_risk = np.clip(0.6 * slope_risk + 0.4 * elev_risk, 0, 1)

    fig = go.Figure(data=go.Heatmap(z=spatial_risk, colorscale='YlOrRd'))
    fig.update_layout(title='Spatial Flood Risk Map (Red = Higher Risk)', height=500)
    st.plotly_chart(fig, use_container_width=True)

def create_raster_overlay(data, colormap):
    """Convert a 2D numpy array to a transparent PNG for folium ImageOverlay."""
    norm = mcolors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap))
    rgba = sm.to_rgba(data, bytes=True)
    img = plt.figure(figsize=(data.shape[1]/100, data.shape[0]/100), dpi=100)
    ax = img.add_axes([0, 0, 1, 1])
    ax.imshow(rgba, aspect='auto')
    ax.axis('off')
    buf = io.BytesIO()
    img.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(img)
    buf.seek(0)
    return buf

def create_map_layers(elevation_data, nasa_data, coordinates):
    st.header("Interactive Map Layers")
    st.markdown("Explore terrain, slope, elevation, and climate data overlaid on the map.")

    south = min(coord[1] for coord in coordinates[0])
    north = max(coord[1] for coord in coordinates[0])
    west = min(coord[0] for coord in coordinates[0])
    east = max(coord[0] for coord in coordinates[0])
    lat_center, lon_center = (south + north) / 2, (west + east) / 2

    # Calculate derived layers
    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    aspect = np.degrees(np.arctan2(-dx, dy))

    # Build available layers
    layers = {
        "Elevation": {"data": elevation_data, "cmap": "terrain", "unit": "meters"},
        "Slope": {"data": slope, "cmap": "YlOrRd", "unit": "degrees"},
        "Aspect": {"data": aspect, "cmap": "hsv", "unit": "degrees"},
    }

    # Slope classification layer
    slope_class = np.zeros_like(slope)
    slope_class[(slope >= 0) & (slope < 5)] = 1    # Flat
    slope_class[(slope >= 5) & (slope < 15)] = 2   # Gentle
    slope_class[(slope >= 15) & (slope < 30)] = 3  # Moderate
    slope_class[slope >= 30] = 4                     # Steep
    layers["Slope Classification"] = {"data": slope_class, "cmap": "RdYlGn_r", "unit": "class"}

    # Buildability layer (flat + moderate elevation = more buildable)
    elev_norm = np.clip((elevation_data - elevation_data.min()) /
                        (elevation_data.max() - elevation_data.min() + 1e-6), 0, 1)
    slope_norm = np.clip(slope / 45, 0, 1)
    buildability = np.clip(1 - 0.6 * slope_norm - 0.4 * elev_norm, 0, 1)
    layers["Buildability"] = {"data": buildability, "cmap": "RdYlGn", "unit": "score (0-1)"}

    # Flood vulnerability from terrain
    slope_risk = np.clip(1 - slope / 30, 0, 1)
    elev_risk = np.clip(1 - (elevation_data - elevation_data.min()) /
                        (elevation_data.max() - elevation_data.min() + 1e-6), 0, 1)
    flood_vuln = np.clip(0.6 * slope_risk + 0.4 * elev_risk, 0, 1)
    layers["Flood Vulnerability"] = {"data": flood_vuln, "cmap": "YlOrRd", "unit": "risk (0-1)"}

    # Solar potential layer (if solar data available)
    if 'ALLSKY_SFC_SW_DWN' in nasa_data.columns:
        avg_solar = nasa_data['ALLSKY_SFC_SW_DWN'].mean()
        # South-facing slopes get more sun in Northern hemisphere
        aspect_factor = np.clip((1 + np.cos(np.radians(aspect - 180))) / 2, 0, 1)
        # Flatter areas and south-facing get better solar
        solar_potential = np.clip(aspect_factor * (1 - slope_norm * 0.3) * (avg_solar / 300), 0, 1)
        layers["Solar Potential"] = {"data": solar_potential, "cmap": "YlOrRd", "unit": "potential (0-1)"}

    # Vegetation suitability (if soil moisture available)
    if 'GWETROOT' in nasa_data.columns:
        avg_moisture = nasa_data['GWETROOT'].mean()
        veg_suit = np.clip(buildability * 0.5 + avg_moisture * 0.5, 0, 1)
        layers["Vegetation Suitability"] = {"data": veg_suit, "cmap": "Greens", "unit": "suitability (0-1)"}

    # WMS-based population layers (added to the raster layers list)
    wms_layers = {
        "Population Density (GPW 2020)": {
            "url": "https://sedac.ciesin.columbia.edu/geoserver/wms",
            "layer": "gpw-v4:gpw-v4-population-density_2020",
        },
        "Population Count (GPW 2020)": {
            "url": "https://sedac.ciesin.columbia.edu/geoserver/wms",
            "layer": "gpw-v4:gpw-v4-population-count_2020",
        },
    }

    all_layer_names = list(layers.keys()) + list(wms_layers.keys())

    # Layer selection
    selected_layers = st.multiselect(
        "Select layers to display on the map",
        all_layer_names,
        default=["Elevation"]
    )

    if not selected_layers:
        st.info("Select at least one layer to display.")
        return

    opacity = st.slider("Layer opacity", 0.1, 1.0, 0.6, 0.05, key="layer_opacity")

    # Create the map
    layer_map = folium.Map(location=[lat_center, lon_center], zoom_start=13)

    # Add different tile layers
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(layer_map)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False
    ).add_to(layer_map)
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap', name='Topographic', overlay=False
    ).add_to(layer_map)

    bounds_rect = [[south, west], [north, east]]

    for layer_name in selected_layers:
        # Raster overlay layers
        if layer_name in layers:
            layer_info = layers[layer_name]
            data = layer_info["data"]
            cmap = layer_info["cmap"]

            buf = create_raster_overlay(data, cmap)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            img_url = f"data:image/png;base64,{img_b64}"

            ImageOverlay(
                image=img_url,
                bounds=bounds_rect,
                opacity=opacity,
                name=f"{layer_name} ({layer_info['unit']})",
                interactive=True,
            ).add_to(layer_map)

        # WMS population layers
        elif layer_name in wms_layers:
            wms_info = wms_layers[layer_name]
            WmsTileLayer(
                url=wms_info["url"],
                layers=wms_info["layer"],
                name=layer_name,
                fmt='image/png',
                transparent=True,
                overlay=True,
                opacity=opacity,
                attr='SEDAC/CIESIN Columbia University'
            ).add_to(layer_map)

    # Draw the selected area boundary
    folium.Polygon(
        locations=[[coord[1], coord[0]] for coord in coordinates[0]],
        color='blue', weight=2, fill=False,
        popup='Selected Area'
    ).add_to(layer_map)

    folium.LayerControl(collapsed=False).add_to(layer_map)
    st_folium(layer_map, width=None, height=550, key="layer_map", use_container_width=True)

    # Layer statistics (only for raster layers, not WMS)
    raster_selected = [l for l in selected_layers if l in layers]
    if raster_selected:
        st.subheader("Layer Statistics")
        stats_cols = st.columns(len(raster_selected))
        for i, layer_name in enumerate(raster_selected):
            data = layers[layer_name]["data"]
            with stats_cols[i]:
                st.markdown(f"**{layer_name}**")
                st.write(f"Min: {np.nanmin(data):.2f}")
                st.write(f"Max: {np.nanmax(data):.2f}")
                st.write(f"Mean: {np.nanmean(data):.2f}")
                st.write(f"Std: {np.nanstd(data):.2f}")

    # Legend explanation
    with st.expander("Layer Descriptions"):
        st.markdown("""
**Terrain Layers (from OpenTopography DEM):**
- **Elevation**: Raw terrain height in meters above sea level
- **Slope**: Terrain steepness in degrees (0° = flat, 45° = very steep)
- **Aspect**: Direction the slope faces (0°/360° = North, 180° = South)
- **Slope Classification**: Flat (green) to Steep (red) terrain categories
- **Buildability**: Combined score — flat, moderate-elevation areas score highest (green)
- **Flood Vulnerability**: Areas at risk based on low elevation + flat terrain (red = higher risk)
- **Solar Potential**: Estimated solar energy potential based on slope orientation and radiation data
- **Vegetation Suitability**: Areas suitable for green spaces based on terrain and soil moisture

**Population Layers (WMS from NASA SEDAC):**
- **Population Density (GPW 2020)**: Census-derived persons per km² at ~1km resolution from Gridded Population of the World v4.11
- **Population Count (GPW 2020)**: Total population count per ~1km grid cell from the same dataset
        """)

def create_urban_features_map(coordinates):
    """Display urban features: population density, roads, amenities, water, green spaces."""
    st.header("Urban Features Map")
    st.markdown("Population density estimation, road networks, amenities, water bodies, and green spaces from OpenStreetMap.")

    south = min(coord[1] for coord in coordinates[0])
    north = max(coord[1] for coord in coordinates[0])
    west = min(coord[0] for coord in coordinates[0])
    east = max(coord[0] for coord in coordinates[0])
    lat_center, lon_center = (south + north) / 2, (west + east) / 2

    with st.spinner("Fetching urban features from OpenStreetMap..."):
        osm_data = get_osm_urban_data(south, north, west, east)

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Buildings", len(osm_data['buildings']))
    col2.metric("Amenities", len(osm_data['amenities']))
    col3.metric("Road Segments", len(osm_data['roads']))
    col4.metric("Water Features", len(osm_data['water']))
    col5.metric("Green Spaces", len(osm_data['green_spaces']))

    # Determine available layers
    available_layers = []
    if osm_data['buildings']:
        available_layers.append("Population Density (Building Heatmap)")
    if osm_data['amenities']:
        available_layers.append("Amenities")
    if osm_data['roads']:
        available_layers.append("Road Network")
    if osm_data['water']:
        available_layers.append("Water Bodies")
    if osm_data['green_spaces']:
        available_layers.append("Green Spaces")

    # WMS-based global population density layers (always available)
    available_layers.append("Population Density - GPW 2020 (Satellite)")
    available_layers.append("Population Count - GPW 2020 (Satellite)")

    if not available_layers:
        st.warning("No urban feature data found for this area. The area may have limited OpenStreetMap coverage.")
        return

    selected_layers = st.multiselect(
        "Select urban feature layers to display",
        available_layers,
        default=available_layers,
        key="urban_layers"
    )

    if not selected_layers:
        st.info("Select at least one layer to display.")
        return

    # Create the map
    urban_map = folium.Map(location=[lat_center, lon_center], zoom_start=13)
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(urban_map)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False
    ).add_to(urban_map)

    # Population density heatmap from building locations
    if "Population Density (Building Heatmap)" in selected_layers and osm_data['buildings']:
        HeatMap(
            osm_data['buildings'],
            name='Population Density',
            radius=15, blur=10, max_zoom=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
        ).add_to(urban_map)

    # Amenities with MarkerCluster
    if "Amenities" in selected_layers and osm_data['amenities']:
        amenity_colors = {
            'hospital': 'red', 'clinic': 'red', 'doctors': 'red', 'pharmacy': 'pink',
            'school': 'blue', 'university': 'darkblue', 'college': 'blue',
            'place_of_worship': 'purple',
            'restaurant': 'orange', 'cafe': 'orange', 'fast_food': 'orange',
            'bank': 'green', 'atm': 'green',
            'police': 'darkred', 'fire_station': 'darkred',
            'fuel': 'gray', 'parking': 'gray',
        }
        cluster = MarkerCluster(name='Amenities').add_to(urban_map)
        for amenity in osm_data['amenities']:
            atype = amenity['type']
            color = amenity_colors.get(atype, 'cadetblue')
            name = amenity['name'] or atype.replace('_', ' ').title()
            folium.Marker(
                location=[amenity['lat'], amenity['lon']],
                popup=f"<b>{name}</b><br>Type: {atype}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(cluster)

    # Road network
    if "Road Network" in selected_layers and osm_data['roads']:
        road_colors = {
            'motorway': '#e74c3c', 'trunk': '#e67e22',
            'primary': '#f39c12', 'secondary': '#3498db',
            'tertiary': '#2ecc71', 'residential': '#95a5a6',
        }
        road_weights = {
            'motorway': 4, 'trunk': 3.5,
            'primary': 3, 'secondary': 2.5,
            'tertiary': 2, 'residential': 1.5,
        }
        road_group = folium.FeatureGroup(name='Road Network')
        for road in osm_data['roads']:
            color = road_colors.get(road['type'], '#bdc3c7')
            weight = road_weights.get(road['type'], 1.5)
            rname = road['name'] or road['type'].replace('_', ' ').title()
            folium.PolyLine(
                road['coords'], color=color, weight=weight,
                opacity=0.8, popup=f"{rname} ({road['type']})"
            ).add_to(road_group)
        road_group.add_to(urban_map)

    # Water bodies
    if "Water Bodies" in selected_layers and osm_data['water']:
        water_group = folium.FeatureGroup(name='Water Bodies')
        for wb in osm_data['water']:
            wname = wb['name'] or wb['type'].replace('_', ' ').title()
            folium.PolyLine(
                wb['coords'], color='#2980b9', weight=3,
                opacity=0.8, popup=f"{wname} ({wb['type']})"
            ).add_to(water_group)
        water_group.add_to(urban_map)

    # Green spaces
    if "Green Spaces" in selected_layers and osm_data['green_spaces']:
        green_group = folium.FeatureGroup(name='Green Spaces')
        for gs in osm_data['green_spaces']:
            gname = gs['name'] or gs['type'].replace('_', ' ').title()
            folium.Polygon(
                gs['coords'], color='#27ae60', fill=True,
                fill_color='#2ecc71', fill_opacity=0.4,
                popup=f"{gname} ({gs['type']})"
            ).add_to(green_group)
        green_group.add_to(urban_map)

    # WMS Population Density layers from SEDAC/NASA
    if "Population Density - GPW 2020 (Satellite)" in selected_layers:
        WmsTileLayer(
            url='https://sedac.ciesin.columbia.edu/geoserver/wms',
            layers='gpw-v4:gpw-v4-population-density_2020',
            name='Population Density (GPW 2020)',
            fmt='image/png',
            transparent=True,
            overlay=True,
            opacity=0.7,
            attr='SEDAC/CIESIN Columbia University'
        ).add_to(urban_map)

    if "Population Count - GPW 2020 (Satellite)" in selected_layers:
        WmsTileLayer(
            url='https://sedac.ciesin.columbia.edu/geoserver/wms',
            layers='gpw-v4:gpw-v4-population-count_2020',
            name='Population Count (GPW 2020)',
            fmt='image/png',
            transparent=True,
            overlay=True,
            opacity=0.7,
            attr='SEDAC/CIESIN Columbia University'
        ).add_to(urban_map)

    # Draw selected area boundary
    folium.Polygon(
        locations=[[coord[1], coord[0]] for coord in coordinates[0]],
        color='blue', weight=2, fill=False, popup='Selected Area'
    ).add_to(urban_map)

    folium.LayerControl(collapsed=False).add_to(urban_map)
    st_folium(urban_map, width=None, height=600, key="urban_features_map", use_container_width=True)

    # Amenity breakdown chart
    if osm_data['amenities']:
        st.subheader("Amenity Distribution")
        amenity_types = [a['type'] for a in osm_data['amenities']]
        type_counts = pd.Series(amenity_types).value_counts().head(15)
        fig = go.Figure(data=[go.Bar(
            x=type_counts.index.str.replace('_', ' ').str.title(),
            y=type_counts.values,
            marker_color='#3498db'
        )])
        fig.update_layout(
            title='Top 15 Amenity Types in Selected Area',
            xaxis_title='Amenity Type', yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Road type breakdown
    if osm_data['roads']:
        st.subheader("Road Network Summary")
        road_types = [r['type'] for r in osm_data['roads']]
        road_counts = pd.Series(road_types).value_counts()
        fig2 = go.Figure(data=[go.Pie(
            labels=road_counts.index.str.replace('_', ' ').str.title(),
            values=road_counts.values,
            hole=0.4
        )])
        fig2.update_layout(title='Road Type Distribution', height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("About Urban Features Data"):
        st.markdown("""
**OpenStreetMap Layers:**
- **Population Density (Building Heatmap)**: Uses building footprint locations from OpenStreetMap as a proxy for population density. Denser building clusters (red) indicate higher population density.
- **Amenities**: Points of interest including hospitals, schools, restaurants, banks, places of worship, etc. Clustered markers expand on zoom.
- **Road Network**: Major road categories color-coded by importance (motorway=red, primary=yellow, secondary=blue, tertiary=green, residential=gray).
- **Water Bodies**: Rivers, streams, canals, and water bodies shown in blue.
- **Green Spaces**: Parks, forests, grasslands, and meadows shown in green.

**Satellite/Census-Based Layers (WMS):**
- **Population Density - GPW 2020**: Gridded Population of the World v4.11 from NASA SEDAC. Shows persons per km² at ~1km resolution based on census data. This is actual census-derived population density, not an estimate.
- **Population Count - GPW 2020**: Total population count per grid cell (~1km) from the same GPW v4.11 dataset.

*OSM data coverage varies by region. GPW data source: [NASA SEDAC/CIESIN Columbia University](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4).*
        """)

def analyze_topography(elevation_data):
    st.subheader("Topography Analysis")

    # Calculate slope
    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))

    # Calculate aspect
    aspect = np.degrees(np.arctan2(-dx, dy))

    # Create visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    im1 = ax1.imshow(elevation_data, cmap='terrain')
    ax1.set_title('Elevation')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')

    im2 = ax2.imshow(slope, cmap='YlOrRd')
    ax2.set_title('Slope')
    plt.colorbar(im2, ax=ax2, label='Slope (degrees)')

    im3 = ax3.imshow(aspect, cmap='hsv')
    ax3.set_title('Aspect')
    plt.colorbar(im3, ax=ax3, label='Aspect (degrees)')

    st.pyplot(fig)

    # Calculate insights
    avg_elevation = np.mean(elevation_data)
    avg_slope = np.mean(slope)

    # Provide insights
    st.subheader("Topography Insights")
    st.write(f"Average Elevation: {np.mean(elevation_data):.2f} meters")
    st.write(f"Average Slope: {np.mean(slope):.2f} degrees")

    # Slope classification
    flat = np.sum((slope >= 0) & (slope < 5)) / slope.size * 100
    gentle = np.sum((slope >= 5) & (slope < 15)) / slope.size * 100
    moderate = np.sum((slope >= 15) & (slope < 30)) / slope.size * 100
    steep = np.sum(slope >= 30) / slope.size * 100

    # Save data to session state
    st.session_state['topography_analysis'] = {
        'average_elevation': avg_elevation,
        'average_slope': avg_slope,
        'slope_classification': {
            'flat': flat,
            'gentle': gentle,
            'moderate': moderate,
            'steep': steep
        }
    }

    st.write("Slope Classification:")
    st.write(f"- Flat (0-5°): {flat:.2f}%")
    st.write(f"- Gentle (5-15°): {gentle:.2f}%")
    st.write(f"- Moderate (15-30°): {moderate:.2f}%")
    st.write(f"- Steep (>30°): {steep:.2f}%")

def create_3d_visualization(elevation_data):
    st.subheader("3D Topography Visualization")
    
    y, x = np.mgrid[0:elevation_data.shape[0], 0:elevation_data.shape[1]]
    
    fig = go.Figure(data=[go.Surface(z=elevation_data, x=x, y=y)])
    
    fig.update_layout(
        title='3D Topography Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Elevation',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=800,
        height=600
    )
    
    st.plotly_chart(fig)

def evaluate_land_suitability(nasa_data, elevation_data, weights=None, ideal_values=None):
    if not isinstance(nasa_data, pd.DataFrame) or not isinstance(elevation_data, np.ndarray):
        raise TypeError("nasa_data must be a pandas DataFrame and elevation_data must be a numpy array")
    
    required_columns = ['T2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN']
    if not all(col in nasa_data.columns for col in required_columns):
        raise ValueError(f"nasa_data must contain columns: {', '.join(required_columns)}")
    
    if weights is None:
        weights = {'temp': 0.3, 'precip': 0.35, 'solar': 0.2, 'slope': 0.15}
    
    if ideal_values is None:
        ideal_values = {'temp': 30, 'precip': 1.5, 'solar': 300, 'slope': 30}
    
    temp_suitability = np.clip(1 - np.abs(nasa_data['T2M'].mean() - ideal_values['temp']) / 30, 0, 1)

    precip_suitability = np.clip(1 - np.abs(nasa_data['PRECTOTCORR'].mean() - ideal_values['precip']) / 1.5, 0, 1)

    solar_suitability = np.clip(nasa_data['ALLSKY_SFC_SW_DWN'].mean() / ideal_values['solar'], 0, 1)

    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    slope_suitability = 1 / (1 + np.exp((slope - ideal_values['slope']) / 10))

    suitability_score = np.clip(
        temp_suitability * weights['temp'] +
        precip_suitability * weights['precip'] +
        solar_suitability * weights['solar'] +
        slope_suitability * weights['slope'],
        0, 1
    )

    return suitability_score

def get_topography_summary(elevation_data):
    dy, dx = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    avg_elevation = np.mean(elevation_data)
    avg_slope = np.mean(slope)
    flat = np.sum((slope >= 0) & (slope < 5)) / slope.size * 100
    gentle = np.sum((slope >= 5) & (slope < 15)) / slope.size * 100
    moderate = np.sum((slope >= 15) & (slope < 30)) / slope.size * 100
    steep = np.sum(slope >= 30) / slope.size * 100
    return {
        'average_elevation': avg_elevation,
        'average_slope': avg_slope,
        'slope_classification': {
            'flat': flat, 'gentle': gentle, 'moderate': moderate, 'steep': steep
        }
    }

def urban_planning_chatbot(nasa_data, elevation_data, user_input, coordinates, chat_history=None):
    avg_temp = nasa_data['T2M'].mean() if 'T2M' in nasa_data.columns else "N/A"
    avg_precip = nasa_data['PRECTOTCORR'].mean() if 'PRECTOTCORR' in nasa_data.columns else "N/A"
    avg_solar = nasa_data['ALLSKY_SFC_SW_DWN'].mean() if 'ALLSKY_SFC_SW_DWN' in nasa_data.columns else "N/A"
    avg_elevation = np.mean(elevation_data)
    lat, lon = calculate_centroid(coordinates)

    topo = get_topography_summary(elevation_data)
    slope_info = topo['slope_classification']

    additional_params = []
    for param in nasa_data.columns:
        if param not in ['T2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN']:
            additional_params.append(f"{ALL_PARAMETERS.get(param, param)}: {nasa_data[param].mean():.2f}")
    additional_climate_str = "\n        ".join(additional_params) if additional_params else "None"

    system_content = f"""You are an advanced AI urban planning assistant with expertise in sustainable development, climate-responsive design, and data-driven decision making.

Available Data:
    Coordinates: Lat {lat:.4f}, Lon {lon:.4f}

    Climate Data:
        Average Temperature: {avg_temp:.2f}°C
        Average Precipitation: {avg_precip:.2f} mm/day
        Average Solar Radiation: {avg_solar:.2f} W/m^2
        {additional_climate_str}

    Topographical Data:
        Average Elevation: {avg_elevation:.2f} meters
        Average Slope: {topo['average_slope']:.2f} degrees
        Flat (0-5°): {slope_info['flat']:.1f}%
        Gentle (5-15°): {slope_info['gentle']:.1f}%
        Moderate (15-30°): {slope_info['moderate']:.1f}%
        Steep (>30°): {slope_info['steep']:.1f}%

Your Capabilities:
    - Analyze the provided data to assess the area's potential for urban development
    - Suggest sustainable urban planning strategies aligned with local climate and topography
    - Recommend optimal building designs, orientations, and ventilation facing
    - Advise on energy-efficient infrastructure, water management, green spaces, transportation, and climate adaptation
    - Identify challenges and suggest mitigation strategies

Response Guidelines:
    - Structure responses clearly with headings and bullet points
    - Provide specific, actionable recommendations referencing the data
    - Explain reasoning behind suggestions
    - State assumptions and limitations clearly"""

    messages = [{"role": "system", "content": system_content}]

    # Add chat history for context (last 10 messages to avoid token limits)
    if chat_history:
        for msg in chat_history[-10:]:
            role = "user" if msg['is_user'] else "assistant"
            messages.append({"role": role, "content": msg['content']})

    messages.append({"role": "user", "content": f"Based on the area data (Lat {lat:.4f}, Lon {lon:.4f}, Avg Temp: {avg_temp:.2f}°C, Avg Precip: {avg_precip:.2f} mm/day, Avg Solar: {avg_solar:.2f} W/m², Avg Elevation: {avg_elevation:.2f}m), {user_input}"})

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def create_chat_interface():
    # Display chat history using st.chat_message
    for message in st.session_state.chat_history:
        role = "user" if message['is_user'] else "assistant"
        with st.chat_message(role):
            st.markdown(message['content'])

    if user_input := st.chat_input("Enter your urban planning query..."):
        st.session_state.chat_history.append({"is_user": True, "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                ai_response = urban_planning_chatbot(
                    st.session_state.nasa_data,
                    st.session_state.elevation_data,
                    user_input,
                    st.session_state.coordinates,
                    st.session_state.chat_history
                )
            st.markdown(ai_response)
        st.session_state.chat_history.append({"is_user": False, "content": ai_response})


# Function to display the welcome page
def show_welcome_page():

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: linear-gradient(45deg, #1a2a6c, #b21f1f, #fdbb2d);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }

        @keyframes gradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .main {
            background-color: rgba(255,255,255,0.9);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        h1 {
            color: #1a2a6c;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            font-weight: 600;
        }

        h3 {
            color: #b21f1f;
            font-weight: 400;
        }

        p, li {
            color: #333;
            line-height: 1.6;
        }

        .stButton>button {
            background-color: #fdbb2d;
            color: #1a2a6c;
            border: none;
            padding: 12px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            font-weight: 600;
            margin: 4px 2px;
            transition-duration: 0.3s;
            cursor: pointer;
            border-radius: 50px;
            box-shadow: 0 4px 15px 0 rgba(252, 104, 110, 0.75);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(252, 104, 110, 0.75);
        }

        /* Animated background shapes */
        .bg-shapes {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }

        .bg-shapes li {
            position: absolute;
            display: block;
            list-style: none;
            width: 20px;
            height: 20px;
            background: rgba(255, 255, 255, 0.2);
            animation: animate 25s linear infinite;
            bottom: -150px;
        }

        .bg-shapes li:nth-child(1) {
            left: 25%;
            width: 80px;
            height: 80px;
            animation-delay: 0s;
        }

        .bg-shapes li:nth-child(2) {
            left: 10%;
            width: 20px;
            height: 20px;
            animation-delay: 2s;
            animation-duration: 12s;
        }

        .bg-shapes li:nth-child(3) {
            left: 70%;
            width: 20px;
            height: 20px;
            animation-delay: 4s;
        }

        @keyframes animate {
            0% {
                transform: translateY(0) rotate(0deg);
                opacity: 1;
                border-radius: 0;
            }
            100% {
                transform: translateY(-1000px) rotate(720deg);
                opacity: 0;
                border-radius: 50%;
            }
        }
    </style>

    <ul class="bg-shapes">
        <li></li>
        <li></li>
        <li></li>
    </ul>
    """, unsafe_allow_html=True)


    st.title("Welcome to the Urban Planning Map Tool")
    
    st.subheader("Overview")
    st.write("""
        This application is designed to assist urban planners in analyzing climate data, topography, and providing urban planning recommendations for specific areas within Pakistan.
        
        *Features:*
        - Visualize climate data including temperature, precipitation, humidity, and more.
        - Analyze topographical features such as elevation and slope.
        - Generate urban planning recommendations based on the analyzed data.
        - Project climate changes over future years.
        
        *Instructions:*
        1. Use the sidebar to select a date range and choose the climate parameters you want to analyze.
        2. On the map, draw a polygon or rectangle to select your area of interest.
        3. Click 'Analyze Selected Area' to fetch and analyze the data.
        4. Explore the visualizations and recommendations provided by the app.
    """)
    
    if st.button("Start Using the App"):
        st.session_state.welcome_done = True
        main()


def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.title("SmartTown: Optimal Land Selection for Urban Planning")

    st.sidebar.header("Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End date", datetime.now()- timedelta(days=3))

    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")
        return

    st.sidebar.subheader("Climate Parameters")
    selected_params = st.sidebar.multiselect(
        "Select climate parameters",
        list(ALL_PARAMETERS.keys()),
        default=list(ALL_PARAMETERS.keys())[:],
        format_func=lambda x: ALL_PARAMETERS[x]
    )

    if not selected_params:
        st.warning("Please select at least one climate parameter.")
        return

    st.subheader("Select Area of Interest")

    default_location = [33.6844, 73.0479]  # Coordinates for Islamabad
    m = folium.Map(location=default_location, zoom_start=12)

    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': True,
            'rectangle': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    map_data = st_folium(m, width=1540, height=450,use_container_width=True)

    if map_data and 'all_drawings' in map_data and map_data['all_drawings'] and len(map_data['all_drawings']) > 0:
        geometry = map_data['all_drawings'][-1]['geometry']
        if geometry['type'] in ['Polygon', 'Rectangle']:
            st.session_state.geometry = geometry
            st.success("Area selected. You can now analyze the data.")
            
            if st.button("Analyze Selected Area",key="selectedArea"):
                coordinates = st.session_state.geometry['coordinates']
                lat, lon = calculate_centroid(coordinates)
                south, west = min(coord[1] for coord in coordinates[0]), min(coord[0] for coord in coordinates[0])
                north, east = max(coord[1] for coord in coordinates[0]), max(coord[0] for coord in coordinates[0])

                area_deg = (north - south) * (east - west)
                if area_deg > 1.0:
                    st.error("Selected area is too large (max ~1 degree x 1 degree / ~110km x 110km). Please select a smaller area to avoid API timeouts.")
                    st.stop()

                progress_bar = st.progress(0)
                
                with st.spinner("Fetching and analyzing data..."):
                    with ThreadPoolExecutor() as executor:
                        nasa_future = executor.submit(get_nasa_power_data, lat, lon, start_date, end_date, selected_params)
                        elevation_future = executor.submit(get_opentopography_data, south, north, west, east)

                        nasa_data = nasa_future.result()
                        elevation_data = elevation_future.result()
                    
                    st.session_state.nasa_data = nasa_data
                    st.session_state.elevation_data = elevation_data
                    st.session_state.coordinates = coordinates
                    
                    progress_bar.progress(100)

                if nasa_data is not None and elevation_data is not None:
                    st.success("Data fetched and analyzed successfully!")
                else:
                    st.error("Failed to fetch or process data. Please try again.")

    if st.session_state.get('nasa_data') is not None and st.session_state.get('elevation_data') is not None:
        nasa_data = st.session_state.nasa_data
        elevation_data = st.session_state.elevation_data
        coordinates = st.session_state.coordinates

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Climate Analysis", "Topography Analysis", "Map Layers",
            "Urban Features", "Land Suitability", "Flood Risk",
            "Area Comparison", "Urban Planning Assistant"
        ])

        with tab1:
            create_climate_visualizations(nasa_data)
            create_monthly_breakdown(nasa_data)
            create_wind_rose(nasa_data)

        with tab2:
            create_3d_visualization(elevation_data)
            analyze_topography(elevation_data)

        with tab3:
            create_map_layers(elevation_data, nasa_data, coordinates)

        with tab4:
            create_urban_features_map(coordinates)

        with tab5:
            st.subheader("Land Suitability Analysis")

            required_columns = ['T2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN']
            missing = [c for c in required_columns if c not in nasa_data.columns]
            if missing:
                missing_names = [ALL_PARAMETERS.get(c, c).split(' - ')[0] for c in missing]
                st.error(f"Land suitability requires these parameters: {', '.join(missing_names)}. Please re-analyze with them selected.")
            else:
                st.subheader("Customize Land Suitability Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Weights (must sum to 1)")
                    weight_temp = st.slider("Temperature Weight", 0.0, 1.0, 0.3, 0.05)
                    weight_precip = st.slider("Precipitation Weight", 0.0, 1.0, 0.35, 0.05)
                    weight_solar = st.slider("Solar Radiation Weight", 0.0, 1.0, 0.2, 0.05)
                    weight_slope = st.slider("Slope Weight", 0.0, 1.0, 0.15, 0.05)
                with col2:
                    st.write("Ideal Values")
                    ideal_temp = st.slider("Ideal Temperature (°C)", 0, 40, 30)
                    ideal_precip = st.slider("Ideal Precipitation (mm/day)", 0.0, 10.0, 1.5, 0.1)
                    ideal_solar = st.slider("Ideal Solar Radiation (W/m^2)", 0, 500, 300, 10)
                    ideal_slope = st.slider("Ideal Slope (%)", 0, 45, 30)

                weight_sum = weight_temp + weight_precip + weight_solar + weight_slope
                if weight_sum == 0:
                    st.error("At least one weight must be greater than 0.")
                else:
                    if abs(weight_sum - 1.0) > 0.01:
                        st.warning(f"Weights sum to {weight_sum:.2f}. They will be normalized to sum to 1.")

                    weights = {
                        'temp': weight_temp / weight_sum,
                        'precip': weight_precip / weight_sum,
                        'solar': weight_solar / weight_sum,
                        'slope': weight_slope / weight_sum
                    }
                    ideal_values = {
                        'temp': ideal_temp,
                        'precip': ideal_precip,
                        'solar': ideal_solar,
                        'slope': ideal_slope
                    }

                    suitability_score = evaluate_land_suitability(nasa_data, elevation_data, weights, ideal_values)
                    fig = go.Figure(data=go.Heatmap(z=suitability_score, colorscale='RdYlGn'))
                    fig.update_layout(title='Land Suitability Heatmap', height=600, width=800)
                    st.plotly_chart(fig)

                    st.write("Suitability Score Legend:")
                    st.write("- Green: More suitable for urban development")
                    st.write("- Yellow: Moderately suitable")
                    st.write("- Red: Less suitable, may require additional considerations")

                    avg_suitability = np.mean(suitability_score)
                    st.metric("Average Land Suitability Score", f"{avg_suitability:.2f}/1.00")

                    # Suitability overlay on map
                    st.subheader("Suitability Overlay on Map")
                    south = min(coord[1] for coord in coordinates[0])
                    north = max(coord[1] for coord in coordinates[0])
                    west = min(coord[0] for coord in coordinates[0])
                    east = max(coord[0] for coord in coordinates[0])
                    lat_center, lon_center = (south + north) / 2, (west + east) / 2

                    suit_map = folium.Map(location=[lat_center, lon_center], zoom_start=13)
                    # Create heatmap data from suitability scores
                    rows, cols = suitability_score.shape
                    heat_data = []
                    for r in range(0, rows, max(1, rows // 50)):
                        for c in range(0, cols, max(1, cols // 50)):
                            lat_pt = south + (north - south) * (rows - r) / rows
                            lon_pt = west + (east - west) * c / cols
                            heat_data.append([lat_pt, lon_pt, float(suitability_score[r, c])])
                    HeatMap(heat_data, radius=15, blur=10, max_zoom=15).add_to(suit_map)
                    st_folium(suit_map, width=700, height=400, key="suitability_map")

                    suitability_df = pd.DataFrame(suitability_score)
                    st.download_button(
                        label="Download Suitability Data as CSV",
                        data=suitability_df.to_csv(index=False),
                        file_name="land_suitability.csv",
                        mime="text/csv"
                    )

        with tab6:
            calculate_flood_risk(nasa_data, elevation_data)

        with tab7:
            st.header("Area Comparison")
            st.markdown("Save the current analysis and compare with other areas.")

            if 'saved_areas' not in st.session_state:
                st.session_state.saved_areas = []

            area_name = st.text_input("Name this area (e.g., 'Site A - Islamabad North')", key="area_name")
            if st.button("Save Current Area for Comparison"):
                if area_name:
                    lat, lon = calculate_centroid(coordinates)
                    topo = get_topography_summary(elevation_data)
                    area_data = {
                        'name': area_name,
                        'lat': lat, 'lon': lon,
                        'avg_temp': nasa_data['T2M'].mean() if 'T2M' in nasa_data.columns else None,
                        'avg_precip': nasa_data['PRECTOTCORR'].mean() if 'PRECTOTCORR' in nasa_data.columns else None,
                        'avg_solar': nasa_data['ALLSKY_SFC_SW_DWN'].mean() if 'ALLSKY_SFC_SW_DWN' in nasa_data.columns else None,
                        'avg_elevation': topo['average_elevation'],
                        'avg_slope': topo['average_slope'],
                        'flat_pct': topo['slope_classification']['flat'],
                    }
                    st.session_state.saved_areas.append(area_data)
                    st.success(f"Area '{area_name}' saved! ({len(st.session_state.saved_areas)} areas saved)")
                else:
                    st.warning("Please enter a name for this area.")

            if len(st.session_state.saved_areas) >= 2:
                comparison_df = pd.DataFrame(st.session_state.saved_areas)
                comparison_df = comparison_df.set_index('name')

                st.subheader("Comparison Table")
                display_cols = {
                    'lat': 'Latitude', 'lon': 'Longitude',
                    'avg_temp': 'Avg Temp (°C)', 'avg_precip': 'Avg Precip (mm/day)',
                    'avg_solar': 'Avg Solar (W/m²)', 'avg_elevation': 'Avg Elevation (m)',
                    'avg_slope': 'Avg Slope (°)', 'flat_pct': 'Flat Area (%)'
                }
                st.dataframe(comparison_df.rename(columns=display_cols).round(2))

                st.subheader("Visual Comparison")
                numeric_cols = ['avg_temp', 'avg_precip', 'avg_solar', 'avg_elevation', 'avg_slope']
                available_cols = [c for c in numeric_cols if c in comparison_df.columns and comparison_df[c].notna().any()]
                if available_cols:
                    compare_param = st.selectbox("Compare parameter", available_cols,
                                                  format_func=lambda x: display_cols.get(x, x))
                    fig = go.Figure(data=[
                        go.Bar(x=comparison_df.index, y=comparison_df[compare_param],
                               marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(comparison_df)])
                    ])
                    fig.update_layout(title=f'{display_cols.get(compare_param, compare_param)} by Area',
                                      yaxis_title=display_cols.get(compare_param, compare_param), height=400)
                    st.plotly_chart(fig, use_container_width=True)

                if st.button("Clear All Saved Areas"):
                    st.session_state.saved_areas = []
                    st.rerun()
            elif len(st.session_state.saved_areas) == 1:
                st.info("1 area saved. Select and analyze a different area on the map, then save it here to compare.")
            else:
                st.info("No areas saved yet. Analyze an area and save it above to start comparing.")

        with tab8:
            st.header("Urban Planning Assistant")
            st.markdown("Discuss your urban planning queries and receive expert advice. The AI remembers your conversation context.")
            create_chat_interface()

        # PDF Report Export
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Report")
        if st.sidebar.button("Generate PDF Report"):
            lat, lon = calculate_centroid(coordinates)
            topo = get_topography_summary(elevation_data)
            slope_info = topo['slope_classification']

            report = io.BytesIO()
            report_text = f"""SMARTTOWN URBAN PLANNING REPORT
{'='*50}

LOCATION
  Coordinates: Lat {lat:.4f}, Lon {lon:.4f}

CLIMATE SUMMARY
"""
            for param in nasa_data.columns:
                param_name = ALL_PARAMETERS.get(param, param).split(' - ')[0]
                report_text += f"  {param_name}: {nasa_data[param].mean():.2f}\n"

            report_text += f"""
TOPOGRAPHY SUMMARY
  Average Elevation: {topo['average_elevation']:.2f} meters
  Average Slope: {topo['average_slope']:.2f} degrees
  Flat (0-5°): {slope_info['flat']:.1f}%
  Gentle (5-15°): {slope_info['gentle']:.1f}%
  Moderate (15-30°): {slope_info['moderate']:.1f}%
  Steep (>30°): {slope_info['steep']:.1f}%

MONTHLY AVERAGES
"""
            monthly = nasa_data.groupby(nasa_data.index.month).mean(numeric_only=True)
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month_idx in monthly.index:
                report_text += f"\n  {month_names[month_idx-1]}:\n"
                for col in monthly.columns:
                    param_name = ALL_PARAMETERS.get(col, col).split(' - ')[0]
                    report_text += f"    {param_name}: {monthly.loc[month_idx, col]:.2f}\n"

            if 'PRECTOTCORR' in nasa_data.columns:
                avg_precip = nasa_data['PRECTOTCORR'].mean()
                heavy_rain = (nasa_data['PRECTOTCORR'] > 10).sum()
                report_text += f"""
FLOOD RISK INDICATORS
  Avg Daily Precipitation: {avg_precip:.2f} mm
  Heavy Rain Days (>10mm): {heavy_rain}
"""

            report_text += f"""
{'='*50}
Generated by SmartTown Urban Planning Tool
"""

            report.write(report_text.encode('utf-8'))
            report.seek(0)
            st.sidebar.download_button(
                label="Download Report",
                data=report,
                file_name="smarttown_report.txt",
                mime="text/plain"
            )

    else:
        st.info("Please draw a polygon or rectangle on the map and click 'Analyze Selected Area' to see the results.")

if __name__ == "__main__":
    if "welcome_done" not in st.session_state:
        st.session_state.welcome_done = False

    if st.session_state.welcome_done:
        main()
    else:
        show_welcome_page()
        