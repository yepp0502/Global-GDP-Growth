import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
import numpy as np

# === Load Data ===
economy = pd.read_csv("economy_updated.csv")
economy['Year'] = economy['Year'].astype(int)
economy['Country'] = economy['Country'].str.strip()
economy['GDP per capita'] = economy['GDP (Millions)'] / economy['Population (Millions)']

# Rename for ISIC consistency
economy.rename(columns={
    'Transport and Communication (Millions)': 'Transport and Communication (ISIC I) (Millions)',
    'Trade and Hospitality (Millions)': 'Trade and Hospitality (ISIC G-H) (Millions)'
}, inplace=True)

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
st.sidebar.header("Go to")
# page = st.sidebar.radio("", ["Overview", "Background", "ISIC Sector", "Additional Insights"])

# === Button State ===
if 'selected' not in st.session_state:
    st.session_state['selected'] = 'Overview'

if st.sidebar.button("Overview", type="primary", use_container_width=True):
  st.session_state['selected'] = 'Overview'
if st.sidebar.button("Background", type="secondary", use_container_width=True):
  st.session_state['selected'] = 'Background'
if st.sidebar.button("ISIC Sector", type="secondary", use_container_width=True):
  st.session_state['selected'] = 'ISIC Sector'
if st.sidebar.button("Additional Insights", type="secondary", use_container_width=True):
  st.session_state['selected'] = 'Additional Insights'

# === Page 1: Overview (Project Info, Boxplot, Objective) ===
if st.session_state['selected'] == 'Overview':
    years = [1970, 1980, 1990, 2000, 2010, 2020]
    subset = economy[economy['Year'].isin(years)]

    box = alt.Chart(subset).mark_boxplot().encode(
        x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('GDP per capita:Q', title='GDP per Capita (USD)', scale=alt.Scale(type='log')),
        color=alt.Color('Year:N', legend=None)
    ).properties(
        title='Distribution of GDP per Capita (1970–2020)',
        width=700,
        height=450
    )

    st.markdown("""
    # 🌍 Global GDP Growth Dashboard

    Welcome to our interactive dashboard exploring macroeconomic trends across over 200 countries from 1970 to 2021. This tool is built to help you investigate the **key drivers of GDP** and understand why some nations grow faster than others.

    ---

    ## 📈 Distribution of GDP per Capita Over Time

    Below is a boxplot showing how **GDP per capita** has changed globally from 1970 to 2020. 
    While the overall median and upper bounds have increased, the growing interquartile range suggests **rising inequality** in global economic prosperity.
    """)

    st.altair_chart(box, use_container_width=True)

    st.markdown("""
    ---

    ## 📌 Project Objective

    This dashboard is part of our exploratory analysis addressing the question:

    > **Why do some countries achieve higher levels of GDP and sustain long-term economic growth, while others fall behind?**

    Through this interface, you can explore:
    - **GDP growth trends**
    - **Sector-specific contributions** to GDP (ISIC categories)
    - The role of **government expenditure**, **trade (imports/exports)**, and **development status**
    - How top-performing countries like **China**, **Equatorial Guinea**, and **South Korea** compare

    ---

    ## 🔍 How to Use This Dashboard

    - Use the **sidebar on the left** to navigate between pages:
      - **Overview**: project intro, global GDP per capita view, and dashboard goals  
      - **Background**: explore top countries by GDP in 2021 and track their long-term GDP trends  
      - **ISIC Sector**: analyze how different industry sectors correlate with GDP
      - **Additional Insights**: examine how trade (imports and exports) and government consumption relate to GDP over time
    
    - All charts are **interactive**:
      - Use sliders and dropdowns to **filter countries by sectors**
      - Hover over points for **tooltips**
      - Click to **highlight and filter** across **linked charts**

    ---

    👉 Use the sidebar to start navigating macroeconomic insights now!
    """)

# === Page 2: Background ===
if st.session_state['selected'] == 'Background':
    st.markdown("""
    # 🌐 Before We Dive Deeper...

    Let’s first understand the background, namely the **top economies in 2021,** and see how their GDP has evolved over the past 50 years.  
    Use the slider below to select how many top countries you'd like to explore, and optionally highlight specific ones for closer inspection.
    """)

    total_countries = economy['Country'].nunique()
    top_n = st.slider("Select number of top countries by GDP (2021):", 5, total_countries, 10)

    latest_data = economy[economy['Year'] == 2021].copy()
    top_countries = latest_data.nlargest(top_n, 'GDP (Millions)')['Country'].tolist()
    filtered = economy[economy['Country'].isin(top_countries)].copy()

    highlight_countries = st.multiselect(
        "Highlight specific countries for deeper comparison:", top_countries,
        default=top_countries[:2]
    )

    highlight_cond = alt.condition(
        alt.FieldOneOfPredicate(field='Country', oneOf=highlight_countries),
        alt.Color('Country:N', legend=None),
        alt.value('lightgray')
    )

    bar_chart = alt.Chart(latest_data[latest_data['Country'].isin(top_countries)]).mark_bar().encode(
        x=alt.X('GDP (Millions):Q', title='GDP in 2021 (Millions USD)'),
        y=alt.Y('Country:N', sort='-x'),
        color=highlight_cond,
        tooltip=['Country', 'GDP (Millions)']
    ).properties(
        title=f'Top {top_n} Countries by GDP in 2021',
        width=400,
        height=400
    )

    line_chart = alt.Chart(filtered).mark_line().encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('GDP (Millions):Q', title='Total GDP (Millions USD)', scale=alt.Scale(type='log')),
        color=highlight_cond,
        tooltip=['Country', 'Year', 'GDP (Millions)']
    ).properties(
        title=f'GDP Trends (1970–2021) for Top {top_n} Countries',
        width=600,
        height=400
    )

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.altair_chart(bar_chart, use_container_width=True)
    with col2:
        st.altair_chart(line_chart, use_container_width=True)

    # === GDP per Capita Rank Improvement ===
    st.markdown("---")
    st.subheader("📈 Largest Improvements in GDP per Capita Rank (1970–2021)")
    st.markdown("""
    The chart below highlights the countries that made the biggest leap in their global **GDP per capita rankings** over the past 50 years.  
    Use the slider to reveal more or fewer countries that achieved notable progress during this time.
    """)

    improve_n = st.slider("Number of countries to display by GDP per capita rank improvement:", 5, 30, 7)

    rank_data = economy[economy['Year'].isin([1970, 2021])].copy()
    rank_data['Rank'] = rank_data.groupby('Year')['GDP per capita'].rank(method='min', ascending=False)

    rank_change = rank_data.pivot(index='Country', columns='Year', values='Rank').dropna().reset_index()
    rank_change['Improvement'] = rank_change[1970] - rank_change[2021]
    rank_change = rank_change.sort_values(by='Improvement', ascending=False).head(improve_n)

    improve_chart = alt.Chart(rank_change).mark_bar().encode(
        x=alt.X('Improvement:Q', title='GDP per Capita Rank Improvement (1970–2021)'),
        y=alt.Y('Country:N', sort='-x'),
        tooltip=['Country', 'Improvement']
    ).properties(
        width=650,
        height=400,
        title=f"Top {improve_n} Countries by GDP per Capita Rank Gain"
    )

    st.altair_chart(improve_chart, use_container_width=True)


# === Page 3: ISIC Sector ===
if st.session_state['selected'] == 'ISIC Sector':
    st.markdown("""
    # 🏣 ISIC Sector Contribution to GDP

    In this section, you can explore the relationship between GDP and sectoral output across different ISIC economic activities.
    Use the selectors below to pick which sectors and countries you'd like to visualize.
    """)

    # --- Minimum GDP threshold ---
    min_gdp_threshold = 50000  # in millions
    economy['GDP per Capita'] = economy['GDP (Millions)'] / economy['Population (Millions)']
    gdp_filter = economy.groupby('Country')['GDP (Millions)'].max() >= min_gdp_threshold
    valid_countries = gdp_filter[gdp_filter].index.tolist()

    st.info(f"ℹ️ **Notice**: Countries with GDP lower than {min_gdp_threshold:,} million USD have been excluded from the dropdown to prevent distorted regression results.")

    # --- Mutually Exclusive Checkboxes ---
    col1, col2 = st.columns(2)
    with col1:
        use_top_gdp = st.checkbox("🔍 Focus on countries with highest GDP in 2021?")
    with col2:
        use_top_growth = st.checkbox("📉 Focus on countries with highest GDP per capita improvement?")

    if use_top_gdp and use_top_growth:
        st.warning("Please select only one focus option at a time.")
        st.stop()

    if use_top_growth:
        pivot = economy.pivot(index='Country', columns='Year', values='GDP per Capita')
        pivot = pivot.dropna(subset=[1970, 2021])
        pivot['Improvement'] = pivot[2021].rank() - pivot[1970].rank()
        top_growth_countries_all = pivot.sort_values(by='Improvement', ascending=False).index.tolist()
        max_growth = len([c for c in top_growth_countries_all if c in valid_countries])
        top_n_growth = st.slider("Select number of countries by GDP per capita rank improvement:", 5, max_growth, 7)
        country_options = [c for c in top_growth_countries_all[:top_n_growth] if c in valid_countries]

    elif use_top_gdp:
        gdp_2021 = economy[economy['Year'] == 2021]
        top_gdp_full = gdp_2021.groupby('Country')['GDP (Millions)'].sum().sort_values(ascending=False)
        top_gdp_countries_all = top_gdp_full.index.tolist()
        max_top = len([c for c in top_gdp_countries_all if c in valid_countries])
        top_n = st.slider("Select number of top countries by GDP to include in dropdown:", 5, max_top, 10)
        country_options = [c for c in top_gdp_countries_all[:top_n] if c in valid_countries]

    else:
        country_options = sorted(valid_countries)

    # --- ISIC dropdowns ---
    isic_columns = {
        "Agriculture (ISIC A-B)": "Agriculture (ISIC A-B) (Millions)",
        "Construction (ISIC F)": "Construction (ISIC F) (Millions)",
        "Manufacturing (ISIC D)": "Manufacturing (ISIC D) (Millions)",
        "Industry (ISIC C-E)": "Industry (ISIC C-E) (Millions)",
        "Transport and Communication (ISIC I)": "Transport and Communication (ISIC I) (Millions)",
        "Trade and Hospitality (ISIC G-H)": "Trade and Hospitality (ISIC G-H) (Millions)",
        "Other Activities (ISIC J-P)": "Other Activities (ISIC J-P) (Millions)"
    }

    selected_sectors = st.multiselect(
        "Choose ISIC sectors to view:", 
        list(isic_columns.keys()), 
        default=["Manufacturing (ISIC D)"]
    )

    selected_countries = st.multiselect(
        "Choose countries:", 
        country_options, 
        default=["China", "United States"] if "China" in country_options and "United States" in country_options else country_options[:2]
    )

    filtered_data = economy[economy['Country'].isin(selected_countries)]

    # --- Store results to compare ---
    slope_summary = {}

    for sector_label in selected_sectors:
        sector_col = isic_columns[sector_label]

        scatter = alt.Chart(filtered_data).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X(f'{sector_col}:Q', title=f'{sector_label} Output (Millions USD)', scale=alt.Scale(zero=False)),
            y=alt.Y('GDP (Millions):Q', title='GDP (Millions USD)', scale=alt.Scale(zero=False)),
            color=alt.Color('Country:N'),
            tooltip=['Country', 'Year', sector_col, 'GDP (Millions)']
        ).properties(
            title=f'GDP vs. {sector_label}',
            width=600,
            height=400
        )

        regression = alt.Chart(filtered_data).transform_regression(
            sector_col, 'GDP (Millions)', groupby=['Country']
        ).mark_line(size=2).encode(
            x=alt.X(f'{sector_col}:Q'),
            y=alt.Y('GDP (Millions):Q'),
            color=alt.Color('Country:N')
        )

        st.altair_chart(scatter + regression, use_container_width=True)

        comparison_results = []
        for country in selected_countries:
            country_df = filtered_data[filtered_data['Country'] == country][[sector_col, 'GDP (Millions)']].dropna()
            if len(country_df) >= 2:
                X = country_df[[sector_col]].values
                y = country_df['GDP (Millions)'].values
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                comparison_results.append((country, slope))

        if comparison_results:
            comparison_results.sort(key=lambda x: abs(x[1]), reverse=True)
            best_country, best_slope = comparison_results[0]
            st.markdown(
                f"📊 **Observation:** Based on regression slope, **{best_country}** shows the strongest relationship between "
                f"**{sector_label}** and GDP *(slope = {best_slope:,.2f})*."
            )
            slope_summary[sector_label] = best_country

    # --- Conclusion ---
    if slope_summary:
        winner_map = {}
        for sector, leader in slope_summary.items():
            winner_map.setdefault(leader, []).append(sector)

        summary_lines = []
        for country, sectors in winner_map.items():
            sectors_list = ", ".join(f"**{s}**" for s in sectors)
            summary_lines.append(f"**{country}** leads in {len(sectors)} sector(s) growth: {sectors_list}")

        st.markdown("---")
        st.markdown("### 🧾 **Sector Leadership Summary**")
        for line in summary_lines:
            st.markdown(line)

# Additional Insights
if st.session_state['selected'] == 'Additional Insights':
    st.markdown("## 🔍 Additional Insights")
    st.markdown("""
    Explore the relationship between \n
      - 📦 **Trade (Imports & Exports)** and \n
      - 🏛️ **Government Consumption** with **GDP**, \n
    using interactive brushing to compare countries.""")

    # Year slider
    selected_year = st.slider(
        "Select a year below to visualize the data:",
        min_value=int(economy['Year'].min()),
        max_value=int(economy['Year'].max()),
        value=2021,
        step=1
    )
    st.markdown(f"Selected Year: **{selected_year}**")

     # Filter data for selected year
    df_year = economy[economy['Year'] == selected_year].copy()
    df_year['Development Status'] = df_year['Per capita GNI'].apply(
        lambda gni: 'Developed' if gni > 13845 else 'Developing'
    )

    # Melt trade data into long format
    trade_data = pd.melt(
        df_year,
        id_vars=["Country", "GDP (Millions)", "Development Status"],
        value_vars=["Exports (Millions)", "Imports (Millions)"],
        var_name="Trade Type",
        value_name="Trade Volume (Millions USD)"
    )

    # Create log columns for regression
    trade_data['log_trade'] = np.log(trade_data['Trade Volume (Millions USD)'])
    trade_data['log_gdp'] = np.log(trade_data['GDP (Millions)'])

    df_year['log_gov'] = np.log(df_year['Government Consumption (Millions)'])
    df_year['log_gdp'] = np.log(df_year['GDP (Millions)'])

    brush = alt.selection_interval(encodings=['y'])

    # Trade vs GDP Chart
    trade_points = alt.Chart(trade_data).mark_circle(size=70, opacity=0.6).encode(
        x=alt.X("Trade Volume (Millions USD):Q", scale=alt.Scale(type='log'), title="Trade Volume (Millions USD)"),
        y=alt.Y("GDP (Millions):Q", scale=alt.Scale(type='log'), title="GDP (Millions USD)"),
        color=alt.condition(
            brush,
            alt.Color("Trade Type:N", scale=alt.Scale(domain=["Imports (Millions)", "Exports (Millions)"], range=["#1f77b4", "#d62728"])),
            alt.value("lightgray")
        ),
        tooltip=["Country", "Trade Type", "Trade Volume (Millions USD)", "GDP (Millions)"]
    ).add_params(brush).properties(
        title="Trade vs GDP",
        width=700,
        height=400
    )

    trade_regression = alt.Chart(trade_data).transform_regression(
        'log_trade', 'log_gdp', groupby=['Trade Type']
    ).transform_calculate(
        exp_x='exp(datum.log_trade)',
        exp_y='exp(datum.log_gdp)'
    ).mark_line(size=2).encode(
        x=alt.X("exp_x:Q", title="Trade Volume (Millions USD)"),
        y=alt.Y("exp_y:Q", title="GDP (Millions USD)"),
        color=alt.Color("Trade Type:N", scale=alt.Scale(domain=["Imports (Millions)", "Exports (Millions)"], range=["#1f77b4", "#d62728"]))
    )

    trade_chart = (trade_points + trade_regression)

    # Government Consumption vs GDP Chart (filtered by Trade Volume brush) ###
    gov_points = alt.Chart(df_year).mark_circle(size=70, opacity=0.6).encode(
        x=alt.X("Government Consumption (Millions):Q", scale=alt.Scale(type='log'), title="Government Consumption (Millions USD)"),
        y=alt.Y("GDP (Millions):Q", scale=alt.Scale(type='log'), title="GDP (Millions USD)"),
        color=alt.Color("Development Status:N", scale=alt.Scale(domain=["Developed", "Developing"], range=["#2ca02c", "#ff7f0e"])),
        tooltip=["Country", "Government Consumption (Millions)", "GDP (Millions)", "Development Status"]
    ).transform_filter(brush).properties(
        title="Government Consumption vs GDP",
        width=700,
        height=400
    )

    gov_regression = alt.Chart(df_year).transform_regression(
    'log_gov', 'log_gdp', groupby=['Development Status']
    ).transform_calculate(
        exp_x='exp(datum.log_gov)',
        exp_y='exp(datum.log_gdp)'
    ).mark_line(size=2).encode(
        x=alt.X("exp_x:Q", title="Gov Consumption (Millions USD)"),
        y=alt.Y("exp_y:Q", title="GDP (Millions USD)"),
        color=alt.Color("Development Status:N", scale=alt.Scale(domain=["Developed", "Developing"], range=["#2ca02c", "#ff7f0e"]))
    )

    gov_chart = (gov_points + gov_regression)

    st.altair_chart(
        alt.vconcat(trade_chart, gov_chart).resolve_scale(
            color='independent'
        ),
        use_container_width=True
    )






