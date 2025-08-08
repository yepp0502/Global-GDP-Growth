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
st.sidebar.markdown("""
<h1 style='text-align: center; margin-bottom: 0.5rem;'>Navigation</h1>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
<h3 style='text-align: center;'>Go to</h3>
""", unsafe_allow_html=True)
# page = st.sidebar.radio("", ["Overview", "Background", "ISIC Sector", "Additional Insights"])

# === Button State ===
if 'selected' not in st.session_state:
    st.session_state['selected'] = 'Overview'

# === Navigation Buttons ===
if st.sidebar.button("Overview", type="primary" if st.session_state['selected'] == 'Overview' else "secondary", use_container_width=True):
    st.session_state['selected'] = 'Overview'

if st.sidebar.button("Background", type="primary" if st.session_state['selected'] == 'Background' else "secondary", use_container_width=True):
    st.session_state['selected'] = 'Background'

if st.sidebar.button("ISIC Sector", type="primary" if st.session_state['selected'] == 'ISIC Sector' else "secondary", use_container_width=True):
    st.session_state['selected'] = 'ISIC Sector'

if st.sidebar.button("Additional Insights", type="primary" if st.session_state['selected'] == 'Additional Insights' else "secondary", use_container_width=True):
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
        title=alt.TitleParams(text = 'Distribution of GDP per Capita (1970–2020)', anchor="middle"),
        width=700,
        height=450
    )

    st.markdown("""
    <style>
    h1, h2 {
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }
    p {
        font-size: 1.1em;
    }
    </style>
    
    # 🌍 **Global GDP Growth Dashboard**
    
    Welcome to our interactive dashboard exploring macroeconomic trends from **1970 to 2021**!  
    
    Goal: Help you investigate the **key drivers of GDP** and **why some nations grow faster than others**.
    
    ---
    
    ## 📈 **Distribution of GDP per Capita Over Time**

    <div style='text-align: center'>
    Here is a boxplot showing how <strong>GDP per capita</strong> has changed globally from 1970 to 2020.
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.altair_chart(box, use_container_width=True)

    st.markdown("Note: As you may see, while the overall median and upper bounds have increased, the growing interquartile range suggests **rising inequality** in global economic prosperity.")

    st.markdown("""
    ---
    
    ## 📌 Dashboard Objective
    
    This dashboard is part of our exploratory analysis addressing the question:
    
    > **Why do some countries grow richer and sustain long-term growth, while others fall behind?**
    
    Through this interface, you can interact with charts and explore:
    - **GDP growth trends** across more than 200 countries  
    - **Sector-specific contributions** to GDP (ISIC categories)  
    - The role of **government expenditure**, **trade (imports/exports)**, and **development status** in GDP levels  
    
    ---
    
    ## 🔍 How to Use This Dashboard?
    
    - Use the **sidebar on the left** to navigate between pages:  
      - **Overview**: Project intro, global GDP per capita preview, and dashboard goals  
      - **Background**: Explore top countries by GDP in 2021 and track their long-term GDP trends  
      - **ISIC Sector**: Analyze how different industry sectors correlate with GDP in various countries  
      - **Additional Insights**: Examine how trade and government consumption relate to GDP over time  

    <br>
    
    - All charts are **interactive**:  
      - Use sliders and dropdowns to **filter countries by sectors**  
      - Hover over points for **tooltips**  
      - Click to **highlight and filter** across **linked charts**  
    """, unsafe_allow_html=True)


# === Page 2: Background ===
if st.session_state['selected'] == 'Background':
    st.markdown("""
        # 🌐Global Status
    
        <div style="font-size: 1.05em;">
          <p>Before we move forward, let’s first understand the top economies in 2021!</p>
        
          <p><strong>Use the slider below to select how many top countries you'd like to explore. You can highlight specific ones for closer inspection as well!</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Cap the number of countries to a maximum of 70
    total_countries = min(economy['Country'].nunique(), 70)
    
    # Left-aligned slider label (default)
    top_n = st.slider("Select number of top countries by GDP (2021):", 5, total_countries, 10)
    
    # Filter top countries
    latest_data = economy[economy['Year'] == 2021].copy()
    top_countries = latest_data.nlargest(top_n, 'GDP (Millions)')['Country'].tolist()
    filtered = economy[economy['Country'].isin(top_countries)].copy()
    
    # Multiselect to highlight countries
    highlight_countries = st.multiselect(
        "Highlight specific countries for deeper comparison:", top_countries,
        default=top_countries[:2]
    )
    
    # Define highlight condition for Altair
    highlight_cond = alt.condition(
        alt.FieldOneOfPredicate(field='Country', oneOf=highlight_countries),
        alt.Color('Country:N', legend=None),
        alt.value('lightgray')
    )
    
    # Bar chart (Top N countries in 2021)
    bar_chart = alt.Chart(latest_data[latest_data['Country'].isin(top_countries)]).mark_bar().encode(
        x=alt.X('GDP (Millions):Q', title='GDP in 2021 (Millions USD)'),
        y=alt.Y('Country:N', sort='-x'),
        color=highlight_cond,
        tooltip=['Country', 'GDP (Millions)']
    ).properties(
        title=alt.TitleParams(text=f'Top {top_n} Countries by GDP in 2021', anchor="middle"),
        width=700,
        height=400
    )
    
    # Line chart (GDP Trends 1970–2021)
    line_chart = alt.Chart(filtered).mark_line().encode(
        x=alt.X('Year:Q', title='Year', scale=alt.Scale(domain=[1970, 2021])),
        y=alt.Y('GDP (Millions):Q', title='Total GDP (Millions USD)', scale=alt.Scale(type='log')),
        color=highlight_cond,
        tooltip=['Country', 'Year', 'GDP (Millions)']
    ).properties(
        title=alt.TitleParams(text=f'GDP Trends (1970–2021) for Top {top_n} Countries', anchor="middle"),
        width=700,
        height=400
    )
    
    # Display stacked charts
    st.altair_chart(bar_chart, use_container_width=True)
    st.altair_chart(line_chart, use_container_width=True)


    # === GDP per Capita Rank Improvement ===
    st.markdown("---")
    st.subheader("📈 Largest Improvements in GDP per Capita Rank")
    st.markdown("""
    <div style="font-size: 1.05em;">
      <p>The chart below highlights the countries that made the biggest growth in GDP per capita rank.</p>
    
      <p><strong>Use the slider to reveal the number of countries that achieved notable progress during this time!</strong></p>
    </div>
    """, unsafe_allow_html=True)

    improve_n = st.slider("Number of countries to display by GDP per capita rank improvement:", 5, 70, 7)

    rank_data = economy[economy['Year'].isin([1970, 2021])].copy()
    rank_data['Rank'] = rank_data.groupby('Year')['GDP per capita'].rank(method='min', ascending=False)

    rank_change = rank_data.pivot(index='Country', columns='Year', values='Rank').dropna().reset_index()
    rank_change['Improvement'] = rank_change[1970] - rank_change[2021]
    rank_change = rank_change.sort_values(by='Improvement', ascending=False).head(improve_n)

    improve_chart = alt.Chart(rank_change).mark_bar().encode(
        x=alt.X('Improvement:Q', title='GDP per Capita Rank Improvement'),
        y=alt.Y('Country:N', sort='-x'),
        tooltip=['Country', 'Improvement']
    ).properties(
        width=650,
        height=400,
        title=alt.TitleParams(text = f"Top {improve_n} Countries by GDP per Capita Rank Improvement",
                             anchor = "middle")
    )
    st.altair_chart(improve_chart, use_container_width=True)


# === Page 3: ISIC Sector ===
if st.session_state['selected'] == 'ISIC Sector':
    st.markdown("""
    # 🏣 ISIC Sector Contribution to GDP
    
    <div style="font-size: 1.05em;">
    
      <p>
        In this section, you can explore the relationship between <strong>GDP</strong> and <strong>sectoral output</strong> across different ISIC economic activities.
      </p>
    
      <p>
        <strong>Use the selectors below to pick which sectors and countries you'd like to visualize.</strong>
      </p>
    
    </div>
    """, unsafe_allow_html=True)

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
        top_n_growth = st.slider("Select number of countries by GDP per capita rank improvement:", 5, 50, 7)
        country_options = [c for c in top_growth_countries_all[:top_n_growth] if c in valid_countries]

    elif use_top_gdp:
        gdp_2021 = economy[economy['Year'] == 2021]
        top_gdp_full = gdp_2021.groupby('Country')['GDP (Millions)'].sum().sort_values(ascending=False)
        top_gdp_countries_all = top_gdp_full.index.tolist()
        max_top = len([c for c in top_gdp_countries_all if c in valid_countries])
        top_n = st.slider("Select number of top countries by GDP to include in dropdown:", 5, 50, 10)
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
            title= alt.TitleParams(
                text=f'GDP vs. {sector_label}',
                anchor="middle"),
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
# === ADDITIONAL INSIGHTS SECTION ===
if st.session_state['selected'] == 'Additional Insights':
    st.markdown("## 🔍 Additional Insights")
    st.markdown("""
    In this section, you will explore the relationship between  
    - 📦 **Trade (Imports & Exports)** with **GDP**,  
    - 🏛️ **Government Consumption** with **GDP**,  
    """)
    st.markdown("""**Use the slider below to explore how trade and government spending relate to GDP over time.**""")

    # === Year slider ===
    selected_year = st.slider(
        "Select a year below to visualize the data:",
        min_value=int(economy['Year'].min()),
        max_value=int(economy['Year'].max()),
        value=2021,
        step=1
    )

    # === Filter data for selected year ===
    df_year = economy[economy['Year'] == selected_year].copy()
    df_year['Development Status'] = df_year['Per capita GNI'].apply(
        lambda gni: 'Developed' if gni > 13845 else 'Developing'
    )

    # === Melt trade data into long format ===
    trade_data = pd.melt(
        df_year,
        id_vars=["Country", "GDP (Millions)", "Development Status"],
        value_vars=["Exports (Millions)", "Imports (Millions)"],
        var_name="Trade Type",
        value_name="Trade Volume (Millions USD)"
    )

    # === Create log-transformed columns ===
    trade_data['log_trade'] = np.log(trade_data['Trade Volume (Millions USD)'].replace(0, np.nan))
    trade_data['log_gdp'] = np.log(trade_data['GDP (Millions)'].replace(0, np.nan))
    df_year['log_gov'] = np.log(df_year['Government Consumption (Millions)'].replace(0, np.nan))
    df_year['log_gdp'] = np.log(df_year['GDP (Millions)'].replace(0, np.nan))

    # === Brushing for Trade chart only ===
    brush = alt.selection_interval(encodings=['y'])

    # === TRADE vs GDP CHART ===
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
        title=alt.TitleParams(
        text="Trade vs GDP",
        anchor="middle"
        ),
        width=700,
        height=400
    )

    trade_regression = alt.Chart(trade_data).transform_regression(
        'log_trade', 'log_gdp', groupby=['Trade Type']
    ).transform_calculate(
        exp_x='exp(datum.log_trade)',
        exp_y='exp(datum.log_gdp)'
    ).mark_line(size=2).encode(
        x="exp_x:Q",
        y="exp_y:Q",
        color=alt.Color("Trade Type:N", scale=alt.Scale(domain=["Imports (Millions)", "Exports (Millions)"], range=["#1f77b4", "#d62728"]))
    )

    trade_chart = trade_points + trade_regression

    # === Calculate slope for Trade chart ===
    slope_summary = {}
    trade_slope_summary = {}
    for ttype in ["Imports (Millions)", "Exports (Millions)"]:
        df_t = trade_data[trade_data['Trade Type'] == ttype].dropna()
        if len(df_t) >= 2:
            X = df_t[['log_trade']].values
            y = df_t['log_gdp'].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            trade_slope_summary[ttype] = slope

    # === DISPLAY: TRADE CHART & OBSERVATION ===
    with st.container():
        st.altair_chart(trade_chart, use_container_width=True)

        if trade_slope_summary:
            sorted_trade = sorted(trade_slope_summary.items(), key=lambda x: abs(x[1]), reverse=True)
            top_trade_type, top_slope = sorted_trade[0]
            st.markdown(f"""
            📈 **Observation**: In **{selected_year}**, **{top_trade_type}** shows the strongest relationship with GDP *(slope = {top_slope:.2f})*
            """)
            slope_summary["Trade"] = top_trade_type

    # === GOVERNMENT CONSUMPTION vs GDP CHART ===
    gov_points = alt.Chart(df_year).mark_circle(size=70, opacity=0.6).encode(
        x=alt.X("Government Consumption (Millions):Q", scale=alt.Scale(type='log'), title="Government Consumption (Millions USD)"),
        y=alt.Y("GDP (Millions):Q", scale=alt.Scale(type='log'), title="GDP (Millions USD)"),
        color=alt.Color("Development Status:N", scale=alt.Scale(domain=["Developed", "Developing"], range=["#2ca02c", "#ff7f0e"])),
        tooltip=["Country", "Government Consumption (Millions)", "GDP (Millions)", "Development Status"]
    ).properties(
        title=alt.TitleParams(
        text="Government Consumption VS GDP",
        anchor="middle"
        ),
        width=700,
        height=400
    )

    gov_regression = alt.Chart(df_year).transform_regression(
        'log_gov', 'log_gdp', groupby=['Development Status']
    ).transform_calculate(
        exp_x='exp(datum.log_gov)',
        exp_y='exp(datum.log_gdp)'
    ).mark_line(size=2).encode(
        x="exp_x:Q",
        y="exp_y:Q",
        color=alt.Color("Development Status:N", scale=alt.Scale(domain=["Developed", "Developing"], range=["#2ca02c", "#ff7f0e"]))
    )

    gov_chart = gov_points + gov_regression

    # === Calculate slope for Gov chart ===
    gov_slope_summary = {}
    for status in ["Developed", "Developing"]:
        df_g = df_year[df_year['Development Status'] == status].dropna(subset=['log_gov', 'log_gdp'])
        if len(df_g) >= 2:
            X = df_g[['log_gov']].values
            y = df_g['log_gdp'].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            gov_slope_summary[status] = slope

    # === DISPLAY: GOV CHART & OBSERVATION ===
    with st.container():
        st.altair_chart(gov_chart, use_container_width=True)

        if gov_slope_summary:
            sorted_gov = sorted(gov_slope_summary.items(), key=lambda x: abs(x[1]), reverse=True)
            top_status, top_gov_slope = sorted_gov[0]
            st.markdown(f"""
            🏛️ **Observation**: In **{selected_year}**, Government consumption in **{top_status}** economies shows stronger correlation with GDP *(slope = {top_gov_slope:.2f})*.
            """)
            slope_summary["Gov"] = top_status

    # === Final Summary Block ===
    if slope_summary:
        st.markdown("---")
        st.markdown(f"### 🧾 **Growth Leadership Summary for {selected_year}**")
        if "Trade" in slope_summary:
            st.markdown(f"- 🌐 **{slope_summary['Trade']}** is the trade factor most strongly associated with GDP.")
        if "Gov" in slope_summary:
            st.markdown(f"- 🏛️ Government consumption in **{slope_summary['Gov']}** economies shows higher GDP responsiveness.")


