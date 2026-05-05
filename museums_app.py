"""
Name:       Your Name
CS230:      Section XXX
Data:       Museums, Aquariums, and Zoos in USA
URL:        (Streamlit Cloud URL if posted)

Description:
    This program explores the Museums, Aquariums, and Zoos dataset for the USA.
    Users can filter institutions by type, state, and revenue using sidebar widgets.
    The app displays an interactive PyDeck map with hover tooltips, bar charts,
    a pie chart, a scatter plot, a pivot table, and a sortable data table.
    All charts use custom colors, titles, and labels for a polished presentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pydeck as pdk

# ── [ST4] Page configuration ───────────────────────────────────────────────────
# st.set_page_config must be the very first Streamlit call in the file.
# layout="wide" gives us more horizontal space.
# initial_sidebar_state="expanded" keeps the sidebar open when the app loads.
st.set_page_config(
    page_title="USA Museums Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── [PY5] Region name dictionary ──────────────────────────────────────────────
# The dataset stores regions as numbers 1-6. This dictionary maps them to names.
# We access its keys and values throughout the app.
REGION_NAMES = {
    1: "New England",
    2: "Mid-Atlantic",
    3: "Southeast",
    4: "Midwest",
    5: "Mountain Plains",
    6: "West",
}

# Simple named colors for charts — one per museum type.
# Matplotlib understands these color names directly, same as using "blue" or "red".
CHART_COLORS = ["steelblue", "tomato", "mediumseagreen", "mediumpurple",
                "darkorange", "lightseagreen", "goldenrod", "slategray", "rosybrown"]

# This is the same format used in class examples.
# searched matplotlib named colors
MAP_COLORS = [
    [70,  130, 180],  # steelblue
    [255, 99,  71],   # tomato
    [60,  179, 113],  # mediumseagreen
    [147, 112, 219],  # mediumpurple
    [255, 140, 0],    # darkorange
    [32,  178, 170],  # lightseagreen
    [218, 165, 32],   # goldenrod
    [112, 128, 144],  # slategray
    [188, 143, 143],  # rosybrown
]


# ── [PY1] Function with a default parameter, called at least twice ─────────────
# n=10 and ascending=False are the default values.
# Called once WITHOUT ascending to get the TOP earners.
# Called once WITH ascending=True to get the LOWEST earners.
def get_top_n(df, column, n=11, ascending=False):
    """
    Return the top (or bottom) n rows of df sorted by column.
    Parameters:
        df        - the dataframe to sort
        column    - the column name to sort by
        n         - how many rows to return (default = 10)
        ascending - sort direction (default = False = largest first)
    """
    # [DA2] Sort data in descending or ascending order
    # [DA3] Find top largest or smallest values of a column
    sorted_df = df.dropna(subset=[column]).sort_values(column, ascending=ascending)
    return sorted_df.head(n)


# ── [PY2] Function that returns more than one value ────────────────────────────
# Python lets you return multiple values at once as a tuple.
# We return mean, median, AND count so the caller can use whichever it needs.
def revenue_stats(df, state=None):
    """
    Calculate revenue statistics for a given state, or all states if state=None.
    Returns: (mean_revenue, median_revenue, count)
    """
    if state:
        subset = df[df["State"] == state]
    else:
        subset = df
    subset = subset.dropna(subset=["Revenue"])
    mean_val   = subset["Revenue"].mean()
    median_val = subset["Revenue"].median()
    count_val  = len(subset)
    return mean_val, median_val, count_val


# ── Load and cache the data ────────────────────────────────────────────────────
# @st.cache_data tells Streamlit to run this function once and remember the result.
# Without it the CSV would reload from scratch every time a slider moves.

# I learned about @st.cache_data from the Streamlit documentation.
# It's a decorator that caches the result of the function so the CSV only loads once,
#which makes the app much faster.

@st.cache_data
def load_data(filepath):
    """Load museums.csv and perform initial cleaning."""

    # [DA1] Clean or manipulate the data
    df = pd.read_csv(filepath, low_memory=False, encoding="utf-8")

    # The column names for city and state are very long — rename them
    df = df.rename(columns={
        "City (Administrative Location)":  "City",
        "State (Administrative Location)": "State",
    })

    # Drop any rows missing latitude or longitude — we cannot plot them on the map
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Replace negative Revenue and Income with NaN — negatives are data errors
    df["Revenue"] = df["Revenue"].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan)
    df["Income"]  = df["Income"].apply( lambda x: x if pd.notna(x) and x >= 0 else np.nan)

    # [DA9] Add a new column: Revenue in millions, rounded to 2 decimal places
    df["Revenue_M"] = (df["Revenue"] / 1_000_000).round(2)

    # [DA7] Add a new Region column by mapping the numeric code using our dictionary
    df["Region"] = df["Region Code (AAM)"].map(REGION_NAMES)

    return df


# ── [PY3] Error handling with try/except ──────────────────────────────────────
# If the CSV is missing we show a friendly error instead of crashing
try:
    df = load_data("C:/Users/dboni/OneDrive - Bentley University/CS230/Museum App/museums.csv")
except FileNotFoundError:
    st.error("Could not find museums.csv.")
    st.stop()



# SIDEBAR WIDGETS


st.sidebar.title("🏛️ Museum Explorer")
st.sidebar.markdown("Use the filters below to explore the data.")
st.sidebar.markdown("---")

# [ST1] Multi-select: user picks one or more museum types
all_types = sorted(df["Museum Type"].unique())
selected_types = st.sidebar.multiselect(
    "🏷️ Select Museum Type(s)",
    options=all_types,
    default=all_types,
)

# [ST2] Selectbox: single dropdown to pick one state or all
all_states = ["All States"] + sorted(df["State"].dropna().unique())
selected_state = st.sidebar.selectbox("📍 Filter by State", options=all_states)

# [ST3] Slider: user sets a minimum and maximum revenue range
# Capped at the 99th percentile so one extreme outlier does not squash the slider
max_rev = int(df["Revenue"].dropna().quantile(0.99))
rev_range = st.sidebar.slider(
    "💰 Revenue Range ($)",
    min_value=0,
    max_value=max_rev,
    value=(0, max_rev),
    step=10_000,
    format="$%d",
)
# $ makes it show $, and %d makes a whole number
st.sidebar.markdown("---")
st.sidebar.caption("Source: Institute of Museum and Library Services (IMLS)")



# APPLY FILTERS


# [DA4] Filter by one condition: keep only selected museum types
filtered = df[df["Museum Type"].isin(selected_types)]

# [DA5] Filter by two conditions with AND: state AND revenue range
if selected_state != "All States":
    filtered = filtered[filtered["State"] == selected_state]

# For revenue charts we drop missing revenue rows then apply the slider range
rev_filtered = filtered.dropna(subset=["Revenue"])
rev_filtered = rev_filtered[
    (rev_filtered["Revenue"] >= rev_range[0]) &
    (rev_filtered["Revenue"] <= rev_range[1])
]



# PAGE HEADER


st.title("🏛️ Museums, Aquariums & Zoos in the USA")
st.markdown(
    f"Exploring **{len(filtered):,}** institutions across the United States. "
    "Use the sidebar to filter by type, state, and revenue."
)
st.markdown("---")

# Call revenue_stats() which returns three values at once [PY2]
mean_rev, median_rev, count_rev = revenue_stats(filtered)

# Four summary numbers across the top of the page
col1, col2, col3, col4 = st.columns(4)
col1.metric("Institutions Shown", f"{len(filtered):,}")
col2.metric("Types Selected",     f"{len(selected_types)}")
col3.metric("Avg Revenue",        f"${mean_rev/1e6:.1f}M" if not np.isnan(mean_rev) else "N/A")
col4.metric("Median Revenue",     f"${median_rev/1e3:.0f}K" if not np.isnan(median_rev) else "N/A")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "📊 Charts", "🏆 Top Institutions", "📋 Data Table"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Geographic Distribution of Institutions")
    st.write(
        "Each dot on the map is one institution. Dots are colored by museum type. "
        "Hover over any dot to see the name, type, city, and revenue."
    )

    # Prepare map data: drop missing coordinates, add a readable revenue string
    map_df = filtered.dropna(subset=["Latitude", "Longitude"]).copy()

    # [PY4] List comprehension: build Revenue_display column in one line
    map_df["Revenue_display"] = [
        f"${r/1e6:.2f}M" if pd.notna(r) else "N/A"
        for r in map_df["Revenue"]
    ]

    # Get the unique museum types in the filtered data
    type_list = sorted(map_df["Museum Type"].unique())

    # Build one sub-dataframe per museum type — same pattern as Boston_Map.py in class
    sub_df_list = []
    for t in type_list:
        sub_df = map_df[map_df["Museum Type"] == t]
        sub_df_list.append(sub_df)

    # Build one ScatterplotLayer per museum type — same pattern as Boston_Map.py in class
    layer_list = []
    for i, sub_df in enumerate(sub_df_list):
        color = MAP_COLORS[i % len(MAP_COLORS)]
        layer = pdk.Layer(
            type="ScatterplotLayer",
            data=sub_df,
            get_position='[Longitude, Latitude]',  # string format same as class
            get_radius=8000,
            get_color=color,                        # [R, G, B] list required by PyDeck
            pickable=True,                          # needed for tooltip to work
            opacity=0.75,
        )
        layer_list.append(layer)

    # Tooltip on hover — same HTML format and orange style as Boston_Map.py in class
    tool_tip = {
        "html": (
            "<b>{Museum Name}</b><br/>"
            "Type: {Museum Type}<br/>"
            "City: {City}, {State}<br/>"
            "Revenue: {Revenue_display}"
        ),
        "style": {
            "backgroundColor": "orange",
            "color": "white",
        },
    }

    # ViewState: starting position and zoom — same structure as class
    view_state = pdk.ViewState(
        latitude=39.5,
        longitude=-98.35,
        zoom=3.5,
        pitch=0,
    )

    # [MAP] Build the map — outdoors-v11 is the same style used in Boston_Map.py
    map = pdk.Deck(
        map_style='mapbox://styles/mapbox/outdoors-v11',
        initial_view_state=view_state,
        layers=layer_list,
        tooltip=tool_tip,
    )

    st.pydeck_chart(map, use_container_width=True)

    # Color legend below the map
    st.markdown("**Color Legend:**")
    legend_cols = st.columns(len(type_list))
    for i, (t, col) in enumerate(zip(type_list, legend_cols)):
        col.markdown(
            f"<span style='background:{CHART_COLORS[i % len(CHART_COLORS)]};padding:2px 6px;"
            f"border-radius:3px;color:white;font-size:11px'>{t}</span>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Data Visualizations")
    st.write("These charts summarize the filtered dataset. Adjust the sidebar filters to see them update.")

    chart_col1, chart_col2 = st.columns(2)

    # ── [VIZ1] Horizontal bar chart: count of institutions per museum type ──
    with chart_col1:
        st.markdown("#### Number of Institutions by Type")
        st.write(
            "Historic Preservation sites are the largest category. "
            "Zoos and Aquariums are the smallest group but tend to earn the most revenue."
        )
        type_counts = filtered["Museum Type"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(6, 4))

        # Use our simple named color list — one color per bar
        bar_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(type_counts))]

        bars = ax1.barh(type_counts.index, type_counts.values,
                        color=bar_colors, edgecolor="white")
        ax1.set_xlabel("Number of Institutions", fontsize=10)
        ax1.set_title("Count by Museum Type", fontsize=12, fontweight="bold")
        ax1.bar_label(bars, fmt="%,.0f", padding=3, fontsize=8)
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax1.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)

    # ── [VIZ2] Pie chart: share of institutions by region ──
    with chart_col2:
        st.markdown("#### Share of Institutions by Region")
        st.write(
            "The Midwest and Southeast together hold nearly 40% of all institutions, "
            "reflecting their dense networks of historic preservation sites."
        )
        region_counts = filtered["Region"].value_counts()

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        wedge_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(region_counts))]

        wedges, texts, autotexts = ax2.pie(
            region_counts.values,
            labels=region_counts.index,
            autopct="%1.1f%%",
            colors=wedge_colors,
            startangle=140,
            pctdistance=0.82,
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax2.set_title("Regional Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    chart_col3, chart_col4 = st.columns(2)

    # ── [VIZ3] Bar chart: average revenue by museum type ──
    with chart_col3:
        st.markdown("#### Average Revenue by Museum Type")
        st.write(
            "Zoos and Aquariums earn far more on average than other types. "
            "Use the revenue slider on the sidebar to focus on different ranges."
        )
        # [DA2] Sort ascending so longest bar appears at the top
        avg_rev = (
            rev_filtered.groupby("Museum Type")["Revenue"]
            .mean()
            .sort_values(ascending=True)
        )

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        bar_colors3 = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(avg_rev))]

        bars3 = ax3.barh(avg_rev.index, avg_rev.values / 1e6,
                         color=bar_colors3, edgecolor="white")
        ax3.set_xlabel("Average Revenue ($ Millions)", fontsize=10)
        ax3.set_title("Average Revenue by Type", fontsize=12, fontweight="bold")
        ax3.bar_label(bars3, fmt="$%.1fM", padding=3, fontsize=8)
        ax3.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)

    # ── [VIZ4] Scatter plot: revenue vs income ──
    with chart_col4:
        st.markdown("#### Revenue vs. Income")
        st.write(
            "Most institutions cluster near zero. Outliers with very high revenue "
            "are large science centers, major zoos, and natural history museums."
        )
        scatter_df = rev_filtered.dropna(subset=["Revenue", "Income"]).copy()
        scatter_df = scatter_df[scatter_df["Income"] >= 0]
        # Cap at 97th percentile so extreme outliers don't squash the chart
        scatter_df = scatter_df[
            (scatter_df["Revenue"] < scatter_df["Revenue"].quantile(0.97)) &
            (scatter_df["Income"]  < scatter_df["Income"].quantile(0.97))
        ]

        fig4, ax4 = plt.subplots(figsize=(6, 4))

        # One set of dots per museum type so each gets its own color and legend entry
        for i, mtype in enumerate(scatter_df["Museum Type"].unique()):
            sub = scatter_df[scatter_df["Museum Type"] == mtype]
            ax4.scatter(
                sub["Revenue"] / 1e6,
                sub["Income"]  / 1e6,
                alpha=0.5,
                s=15,
                color=CHART_COLORS[i % len(CHART_COLORS)],
                label=mtype,
            )

        ax4.set_xlabel("Revenue ($ Millions)", fontsize=10)
        ax4.set_ylabel("Income ($ Millions)",  fontsize=10)
        ax4.set_title("Revenue vs. Income", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=6, loc="upper left", framealpha=0.6)
        ax4.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)

    # ── [DA6] Pivot table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Pivot Table: Average Revenue by State and Museum Type")
    st.write(
        "Shows mean revenue (in $M) for the five most common museum types by state. "
        "Blank cells mean no data exists for that combination. Darker blue = higher revenue."
    )

    top5_types = df["Museum Type"].value_counts().head(5).index.tolist()

    pivot_df = (
        rev_filtered[rev_filtered["Museum Type"].isin(top5_types)]
        .pivot_table(
            index="State",
            columns="Museum Type",
            values="Revenue",
            aggfunc="mean",
        )
        / 1e6
    ).round(2)

    st.dataframe(
        pivot_df.style
            .format("${:.2f}M", na_rep="—")
            .background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
        height=300,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOP INSTITUTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏆 Top & Bottom Institutions by Revenue")
    st.write("Use the slider to choose how many institutions to display in each table.")

    n_top = st.slider("How many institutions to show?", min_value=5, max_value=25, value=10)

    col_top, col_bot = st.columns(2)

    # FIRST call of get_top_n — uses the DEFAULT ascending=False [PY1]
    with col_top:
        st.markdown("#### 🥇 Highest Revenue")
        top_n = get_top_n(filtered, "Revenue", n=n_top)
        display_top = top_n[["Museum Name", "Museum Type", "State", "Revenue"]].copy()
        display_top["Revenue"] = display_top["Revenue"].apply(lambda x: f"${x/1e6:.2f}M")
        st.dataframe(display_top.reset_index(drop=True), use_container_width=True)

    # SECOND call of get_top_n — passes ascending=True explicitly [PY1]
    with col_bot:
        st.markdown("#### 📉 Lowest Revenue (above $0)")
        positive_rev = filtered[filtered["Revenue"] > 0]
        bot_n = get_top_n(positive_rev, "Revenue", n=n_top, ascending=True)
        display_bot = bot_n[["Museum Name", "Museum Type", "State", "Revenue"]].copy()
        display_bot["Revenue"] = display_bot["Revenue"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_bot.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Top 15 States by Total Revenue")
    st.write(
        "California, New York, and Illinois consistently lead in total reported revenue, "
        "driven by large metropolitan science museums and art institutions."
    )

    # [DA8] Iterate through rows with iterrows() to build a total revenue per state
    # iterrows() returns (index, row) pairs — same as taught in class
    state_totals = {}
    for _, row in rev_filtered.iterrows():
        state = row["State"]
        rev   = row["Revenue"]
        if pd.notna(rev):
            state_totals[state] = state_totals.get(state, 0) + rev

    state_series = pd.Series(state_totals).sort_values(ascending=False).head(15)

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    bar_colors5 = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(state_series))]

    bars5 = ax5.bar(state_series.index, state_series.values / 1e9,
                    color=bar_colors5, edgecolor="white")
    ax5.set_ylabel("Total Revenue ($ Billions)", fontsize=10)
    ax5.set_title("Top 15 States by Total Museum Revenue", fontsize=13, fontweight="bold")
    ax5.bar_label(bars5, fmt="$%.1fB", padding=2, fontsize=8)
    ax5.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Filtered Data Table")
    st.write(
        f"Showing **{len(filtered):,}** institutions matching your current filters. "
        "Click any column header to sort. Use the sidebar to change the filters."
    )

    display_cols = ["Museum Name", "Museum Type", "City", "State",
                    "Region", "Revenue_M", "Income", "Latitude", "Longitude"]
    show_df = filtered[display_cols].copy()
    show_df = show_df.rename(columns={"Revenue_M": "Revenue ($M)"})

    st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=450)

    # Download button: saves the filtered data as a CSV file
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=show_df.to_csv(index=False).encode("utf-8"),
        file_name="museums_filtered.csv",
        mime="text/csv",
    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: Institute of Museum and Library Services (IMLS) · CS 230 Final Project · Spring 2026")
