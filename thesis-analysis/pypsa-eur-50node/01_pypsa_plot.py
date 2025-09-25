"""
50-Node Case Study: Basic Network Visualization
------------------------------------------------

This script prepares a PyPSA-Eur 50-node case for analysis in `munacco`
and visualizes:

1. **Geographical map of the network**
   - Zones are filled with distinct colors
   - Transmission lines are colored and scaled by capacity (f_max)
   - Nodes shown as black-rimmed white markers
   - Interactive Plotly map (Robinson projection)

2. **Stacked capacity comparison by zone**
   - Comparing low vs. high RES cases
   - Carriers grouped and color-coded
   - Subplots per bidding zone with shaded zone backgrounds

Purpose:
Provide an overview of the system topology and differences in
installed capacities across RES penetration levels.

"""

import munacco as mc
import pypsa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import pycountry
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


# ----------------------------
# Step 1: Load PyPSA-Eur 50 node network
# ----------------------------
n = pypsa.Network("thesis-data/pypsa-eur-50node/cwe_network_data/solved_network/base_s_50_elec_.nc")
curtailable_share = {
    "offwind-dc": 0.7,
    "offwind-ac": 0.7,
    "onwind": 0.7,
    "solar": 0.5
}

loader = mc.InputLoader()
network = loader.load_from_pypsa(network=n, snapshot=1, initialize=False, p_nom_opt=True)
network.res['sigma'] = 0.1
network.split_res_generators(curtailable_share)

# Adjust generation and load balance
network.plants.loc[network.plants.g_max < 800, 'alpha'] = False
network.lines['f_max'] = np.maximum(network.lines['f_max'] * 0.7, 400)

# Adjust loads to avoid infeasible cases (scale if RES > load)
for z in network.zones.index:
    n_in_zone = network.nodes[network.nodes.zone == z].index
    g_in_zone = sum(network.plants.loc[network.plants.bus.isin(n_in_zone), 'g_max']
                    * network.plants.loc[network.plants.bus.isin(n_in_zone), 'p_max_pu'])
    res_in_zone = sum(network.res.loc[network.res.bus.isin(n_in_zone), 'g_max']
                      * network.res.loc[network.res.bus.isin(n_in_zone), 'p_max_pu'])
    res_fix_in_zone = sum(network.res.loc[network.res.bus.isin(n_in_zone) & ~network.res.RD, 'g_max']
                          * network.res.loc[network.res.bus.isin(n_in_zone) & ~network.res.RD, 'p_max_pu'])
    load_in_zone = network.nodes.loc[network.nodes.zone == z, 'Pd'].sum()

    faktor1 = (g_in_zone + res_in_zone) / load_in_zone
    if faktor1 < 1:
        network.nodes.loc[network.nodes.zone == z, 'Pd'] *= faktor1 * 0.99

    if res_fix_in_zone > 0:
        faktor2 = load_in_zone / res_fix_in_zone
        if faktor2 < 1:
            network.nodes.loc[network.nodes.zone == z, 'Pd'] *= 1 / (faktor2 * 0.99)

network.initialize()

# ----------------------------
# Step 2: Zone color definitions
# ----------------------------
zone_colors = {
    'AT': "#f0f8ff",  # light blue
    'BE': "#fff5e6",  # light orange
    'DE': "#d0cafc",  # light gray
    'FR': "#dbcfba",  # light green
    'NL': "#ffe6f0",  # light pink
    'LU': "#d0cafc",  # light gray
}

# ----------------------------
# Step 3: Helper Functions
# ----------------------------
def alpha2_to_alpha3(alpha2):
    """Convert ISO2 code to ISO3 code (fallback=None)."""
    try:
        return pycountry.countries.get(alpha_2=alpha2).alpha_3
    except:
        return None


def plot_network_map(network, country_shapes=None):
    """
    Plot the network on a geographic map:
    - Countries shaded by zone_colors
    - Transmission lines scaled by f_max
    - Nodes shown as markers
    """

    # --- Country shapes ---
    if country_shapes is None:
        gdf = gpd.read_file(
            "test_pypsa/network_data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
        )
        country_shapes = gdf[gdf["REGION_UN"] == "Europe"]

    zones2 = network.nodes["zone"].unique()
    zones3 = [alpha2_to_alpha3(z) for z in zones2 if alpha2_to_alpha3(z)]
    zones3.append('LUX')
    iso3_to_iso2 = {alpha2_to_alpha3(z): z for z in zones2 if alpha2_to_alpha3(z)}
    iso3_to_iso2['LUX'] = 'LU'

    fig = go.Figure()

    # --- Add polygons for each zone ---
    def add_polygon(x, y, color, zone_code):
        fig.add_trace(go.Scattergeo(
            lon=list(x), lat=list(y),
            mode="lines",
            line=dict(color="black", width=2),
            fill="toself", fillcolor=color,
            name=zone_code,
            showlegend=False
        ))

    for _, row in country_shapes.iterrows():
        zone_code = row["GU_A3"]
        if zone_code not in zones3:
            continue
        color = (zone_colors[iso3_to_iso2[zone_code]])
        geom = row["geometry"]
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            add_polygon(x, y, color, zone_code)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                add_polygon(x, y, color, zone_code)

    # --- Legend: dummy traces for zones ---
    for z2, rgba in zone_colors.items():
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode="markers",
            marker=dict(size=15, color=rgba, line=dict(width=1, color="black"), symbol="square"),
            name=z2
        ))

    # --- Transmission lines scaled by f_max ---
    fmax_values = network.lines["f_max"].astype(float)
    vmin, vmax = fmax_values.min(), fmax_values.max()
    colorscale = px.colors.sequential.Agsunset
    min_w, max_w = 1, 10

    for _, line in network.lines.iterrows():
        ni, nj = line["node_i"], line["node_j"]
        xi, yi = network.nodes.loc[ni, ["x", "y"]]
        xj, yj = network.nodes.loc[nj, ["x", "y"]]

        val = ((line["f_max"] - vmin) / (vmax - vmin))
        color = sample_colorscale(colorscale, val)[0]
        width = min_w + (max_w - min_w) * val

        fig.add_trace(go.Scattergeo(
            lon=[xi, xj], lat=[yi, yj],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=0.9,
            showlegend=False
        ))

    # Add colorbar
    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None], mode="markers",
        marker=dict(
            size=0.1, color=fmax_values/1000, colorscale=colorscale, showscale=True,
            colorbar=dict(
                title=dict(text="Line Capacity [GW]", side="right", font=dict(size=30)),
                x=0.75, y=0.5, len=0.7
            )
        ),
        showlegend=False
    ))

    # --- Nodes ---
    fig.add_trace(go.Scattergeo(
        lon=network.nodes["x"], lat=network.nodes["y"],
        text=network.nodes.index,
        mode="markers",
        marker=dict(size=10, color="white", line=dict(width=2, color="black")),
        textposition="top center",
        name="Nodes", showlegend=False
    ))

    # --- Layout ---
    lonmin, lonmax = network.nodes["x"].min()-2, network.nodes["x"].max()+2
    latmin, latmax = network.nodes["y"].min()-2, network.nodes["y"].max()+2

    fig.update_layout(
        geo=dict(
            projection_type="robinson",
            lonaxis=dict(range=[lonmin, lonmax]),
            lataxis=dict(range=[latmin, latmax]),
            showland=False, showcountries=False, showcoastlines=False,
            showocean=False, showframe=False, bgcolor="rgba(0,0,0,0)"
        ),
        width=2000, height=800,
        font=dict(family="Serif", size=28),
        margin=dict(l=20, r=80, t=20, b=20),
        legend=dict(orientation="v", yanchor="bottom", y=0.6, xanchor="center", x=0.3),
        legend_tracegroupgap=10
    )
    return fig


# ----------------------------
# Step 4: Plot map
# ----------------------------
fig = plot_network_map(network)
fig.show()

#%%
# ----------------------------
# Step 5: Compare Low vs. High RES capacity mixes
# ----------------------------
carrier_colors = {
    "CCGT": "#d73027", "OCGT": "#fc8d59", "coal": "#252525", "lignite": "#636363",
    "oil": "#8c510a", "nuclear": "#762a83", "biomass": "#1a9850", "ror": "#4575b4",
    "onwind": "#91bfdb", "offwind-ac": "#2c7bb6", "offwind-dc": "#313695", "solar": "#ffd700"
}

def get_cap_by_zone(network):
    """Return total installed capacity per (carrier, zone)."""
    plants_info = network.plants[["carrier", "g_max", "zone"]].copy()
    res_info = network.res[["carrier", "g_max", "zone"]].copy()
    return pd.concat([plants_info, res_info]).groupby(["carrier", "zone"])["g_max"].sum().unstack(fill_value=0)

# Load low vs. high RES cases
network_low  = loader.load_from_pypsa(network=n, snapshot=1, initialize=False, p_nom_opt=False)
network_high = loader.load_from_pypsa(network=n, snapshot=1, initialize=False, p_nom_opt=True)
network_high.res.loc['AT0 1 0 solar', 'g_max'] /= 2
network_high.res.loc['AT0 2 0 solar', 'g_max'] /= 2

for net in [network_low, network_high]:
    node_zone_map = {node: net.Z[row.argmax()] for node, row in zip(net.N, net.map_nz)}
    net.plants['zone'] = net.plants.bus.map(node_zone_map)
    net.res['zone']    = net.res.bus.map(node_zone_map)

cap_low  = get_cap_by_zone(network_low) / 1000
cap_high = get_cap_by_zone(network_high) / 1000

zones, carriers = cap_low.columns, cap_low.index
ymax = max(cap_low.sum().max(), cap_high.sum().max()) * 1.05

fig = make_subplots(rows=1, cols=len(zones),
    subplot_titles=[f"{z}" for z in zones],
    shared_yaxes=True, horizontal_spacing=0.019)

for i, zone in enumerate(zones, start=1):
    for carrier in carriers:
        fig.add_trace(go.Bar(
            x=["<b>Low RES</b>", "<b>High RES</b>"],
            y=[cap_low.loc[carrier, zone], cap_high.loc[carrier, zone]],
            name=carrier,
            marker_color=carrier_colors.get(carrier, "black"),
            showlegend=(i == 1),
        ), row=1, col=i)

fig.update_layout(
    barmode="stack", bargap=0,
    font=dict(family="Serif", size=50),
    plot_bgcolor="white",
    legend=dict(title="", orientation="h", font=dict(size=30), y=-0.15),
    width=1400, height=600,
    margin=dict(l=40, r=40, t=80, b=80)
)
fig.update_annotations(font_size=35)

for i, zone in enumerate(zones, start=1):
    fig.update_xaxes(row=1, col=i,
        showgrid=False, showline=True, linewidth=2, linecolor="black",
        ticks="outside", showticklabels=True, tickfont=dict(size=20))
    fig.update_yaxes(row=1, col=i, range=[0, ymax],
        showgrid=True, gridcolor="lightgray", griddash="dot",
        showline=True, linewidth=2, linecolor="black",
        ticks="outside", showticklabels=(i == 1),
        title_text="Capacity [GW]" if i == 1 else None,
        title_font=dict(size=35), tickfont=dict(size=35))

# Add shaded zone backgrounds
for i, zone in enumerate(zones, start=1):
    xref = f"x{i} domain" if i > 1 else "x domain"
    yref = f"y{i} domain" if i > 1 else "y domain"
    fig.add_shape(
        type="rect", xref=xref, yref=yref,
        x0=-0.35 if i==1 else -0.05, x1=1.05,
        y0=-0.13, y1=1.15,
        fillcolor=zone_colors.get(zone, "#ffffff"), opacity=0.7,
        line=dict(color="black", width=2), layer="below"
    )

fig.show()
