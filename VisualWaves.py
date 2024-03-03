import streamlit as st
import pandas as pd
import altair as alt
from vega_datasets import data
import matplotlib.pyplot as plt

@st.cache
def load_data():
    country_df = pd.read_csv('https://raw.githubusercontent.com/hms-dbmi/bmi706-2022/main/cancer_data/country_codes.csv', dtype={'country-code': str})
    df = pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/country.csv')
    df['totaltrials'] = df.groupby(['Study population', 'year', 'phase'])['Study population'].transform('count')
    pharma = pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/pharma_country.csv', encoding='latin1')
    pharma2=pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/minus_OLE_with_generalized_indications.csv')


    merged_df = pd.merge(df, country_df[['ID', 'country-code']], left_on='Study population', right_on='Country', how='left').dropna()
    merged_df = merged_df.dropna()
    merged_df['year'] = merged_df['year'].astype(int)
    merged_pharma = pd.merge(pharma, country_df[['Country', 'country-code']], left_on='Study population', right_on='Country', how='left')

    race_df_ol=pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/age_ol.csv')
    race_df_rct=pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/age_rct.csv')
    combined_race_df = pd.concat([race_df_ol, race_df_rct], axis=0)
   

df = load_data()


st.set_page_config(layout="wide")
# Streamlit app layout
st.title('Clinical Trials Explorer')

# Selector for choosing between different themes
selected_theme = st.sidebar.selectbox("Select Dashboard", ["Country", "Funding", "Demographics"])

# Common selectors for year and phase
selected_year = st.sidebar.slider('Select Year', min_value=min(merged_df['year']), max_value=max(merged_df['year']), value=(min(merged_df['year']), max(merged_df['year'])))
selected_phases = st.sidebar.multiselect('Select Phase(s)', options=merged_df['phase'].unique(), default=merged_df['phase'].unique())

# Filter data based on selected year and phase
df_filtered_by_phase = merged_df[(merged_df['year'].between(selected_year[0], selected_year[1])) & (merged_df['phase'].isin(selected_phases))]
pharma2_filtered_by_phase= pharma2[(pharma2['year'].between(selected_year[0], selected_year[1])) & (pharma2['phase'].isin(selected_phases))]




if selected_theme == "Country":
    left_column, right_column = st.columns([5, 10])
    with left_column:
    # Country Ranking List
        st.subheader('Number of trials by country')

        country_rank = df_filtered_by_phase.groupby('Study population')['totaltrials'].sum().reset_index().sort_values('totaltrials', ascending=False)
        for _, row in country_rank.iterrows():
            total_trials = int(row['totaltrials'])
            st.markdown(f'<p class="custom-text">{row["Study population"]}: {total_trials}</p>', unsafe_allow_html=True)

    with right_column:
    # Geospatial Chart
        st.subheader('Geospatial Distribution')
    # Vega_datasets world data
        source = alt.topo_feature(data.world_110m.url, 'countries')
        width = 600
        height  = 500
        project = 'equirectangular'

    # a gray map using as the visualization background
        background = alt.Chart(source
        ).mark_geoshape(
            fill='#aaa',
            stroke='white'
        ).properties(
            width=width,
            height=height
        ).project(project)
    
        aggregated_data = df_filtered_by_phase.groupby(['Study population','country-code'])['totaltrials'].sum().reset_index()
        pharma_total = merged_pharma.groupby(['Study population', 'country-code']).size().reset_index(name='count')

        selector = alt.selection_single(fields=['Study population'], on='click', empty="all", clear='dblclick')

        chart_base = alt.Chart(source
        ).properties(
            width=width,
            height=height
        ).project(project
        ).add_selection(selector
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(aggregated_data, "country-code", ["Study population","totaltrials"]),
        )

    
        total_trials_sum = aggregated_data['totaltrials'].max()
        rate_scale = alt.Scale(domain=[0, total_trials_sum], scheme='oranges')
        rate_color = alt.Color(field="totaltrials", type="quantitative", scale=rate_scale)

        chart_rate = chart_base.mark_geoshape().encode(
            color=rate_color,
            tooltip=[
                alt.Tooltip('Study population:N', title='Country'),
                alt.Tooltip('totaltrials:Q')
            ]
        ).transform_filter(selector
        ).properties(
            title=f'Number of Trials by country'
        )

        pharma_chart_base = alt.Chart(source
        ).properties(
            width=width,
            height=height
        ).project(project
        ).add_selection(selector
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(pharma_total, "country-code", ["Study population","count"]),
        )

        pharma_trials_sum = pharma_total['count'].max()
        pharma_rate_scale = alt.Scale(domain=[0, pharma_trials_sum], scheme='yellowgreenblue')
        pharma_rate_color = alt.Color(field="count", type="quantitative", scale=pharma_rate_scale)

        pharma_chart_rate = pharma_chart_base.mark_geoshape().encode(
            color=pharma_rate_color,
            tooltip=[
                alt.Tooltip('Study population:N', title='Country'),
                alt.Tooltip('count:Q', title='Number of funding sources by country')
            ]
        ).transform_filter(selector
        ).properties(
            title=f'Number of Funding Sources by country'
        )

        chart2 = alt.vconcat(background + chart_rate, background + pharma_chart_rate
        ).resolve_scale(
            color='independent'
        )

        st.altair_chart(chart2)

        # Country Selector
        country_trials = df_filtered_by_phase.groupby('Study population')['totaltrials'].sum()
        top_countries = country_trials.sort_values(ascending=False).index[:20]
        country = st.selectbox('Select Country', options=top_countries)
        df_country = df_filtered_by_phase[df_filtered_by_phase['Study population'] == country]

        # Line and Dot Graph for the selected country
        st.subheader(f'Number of Trials from {selected_year[0]} to {selected_year[1]} for {country}')
        line_chart = alt.Chart(df_country).mark_line(point=True).encode(
            x='year:O',
            y=alt.Y('totaltrials:Q',axis=alt.Axis(title='Count')),
            color='phase:O',
            tooltip=['year', 'phase', 'totaltrials']
        ).properties(
            width=500,
            height=500
        ).configure_axis(
            gridOpacity=0
        )


        st.altair_chart(line_chart, use_container_width=True)


 
elif selected_theme == "Funding":
    # Display charts related to funding theme
    st.subheader('Funding')
    
    # Add your funding-related charts here
    st.write("Charts related to Funding theme")
        #pie_chart for funding source
    st.subheader(f'Total trials over years by top 10 funding sources')
    pharma_selection = alt.selection_single(fields=['source'],bind='legend',on='click',empty="all",clear='dblclick')
    funding=pharma2_filtered_by_phase.groupby(['source']).size().reset_index(name='count')
    funding_sorted = funding.sort_values(by='count', ascending=False)
    top_10_funding = funding_sorted.head(10)
    pie_chart = alt.Chart(top_10_funding).mark_arc().encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field="source", type="nominal"),
        tooltip=['source', 'count']
    ).add_selection(
        pharma_selection
    )

    company_summary = pharma2_filtered_by_phase.groupby(['source', 'year']).size().reset_index(name='count')
    
    # Create the line chart with filtered data based on the selection
    line_chart = alt.Chart(company_summary).mark_line(point=True).encode(
        x=alt.X('year:O'),
        y='count:Q',
        color='source:N',
        tooltip=['source', 'year', 'count']
    ).transform_filter(
        pharma_selection
    ).interactive()

    # Display the combined chart
    combined_chart = pie_chart | line_chart
    st.altair_chart(combined_chart, use_container_width=True)
    # Add additional charts or data related to funding theme here
    st.title('Seizure Type Comparison Across Age Groups')

    # Load the dataset from the specified path
    data = pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/minus_OLE_with_generalized_indications_age_groups.csv')

    # Filter for the specified seizure types
    seizure_types_of_interest = ['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status']
    filtered_data = data[data['indication_gen'].isin(seizure_types_of_interest)]

    # Aggregate the data by 'Age Group' and 'indication_gen'
    aggregated_data = filtered_data.groupby(['Age Group', 'indication_gen']).size().unstack(fill_value=0)


    # Plotting
    fig, ax = plt.subplots()
    aggregated_data.plot(kind='bar', figsize=(10, 7), ax=ax)
    plt.title('Comparison of Seizure Types Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Seizure Type')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    



elif selected_theme == "Demographics":
     st.subheader('Demographics')

     df_race = pd.merge(df_filtered_by_phase["ID", "year", 'source', 'phase'], combined_race_df, on='ID', how='left').melt( 
     id_vars=["ID", "year", 'phase','source',],
     var_name="Race",
     value_name="participants_race",)
 
     df_gender = df_filtered_by_phase["ID", "year", 'source', 'phase', "Male", "Female"].melt( 
     id_vars=["ID", "year", 'phase','source',],
     var_name="Gender",
     value_name="participants_gender",)

     df_dem = pd.merge(df_race, df_gender, on=['ID', "year", 'source', 'phase',], how='left')


     num_years= df_race['year'].nunique()

     if num_years > 10:
         year_bins = np.linspace(df_dem['year'].min(), df_dem['year'].max(), num=11)
         df_dem['Year_Range'] = pd.cut(df_dem['year'], bins=year_bins, include_lowest=True)
         df_dem['Year_Range'] = df_dem['Year_Range'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
     else:
         df_dem['Year_Range'] = df_dem['year'].astype(str)

     df_dem['NormalizedValueRace'] = (df_dem.groupby(['Year_Range', 'Race'])['participants_race'].transform(lambda x: (x / x.sum())*100))

     base = alt.Chart(df_dem
                     ).transform_aggregate(
         total_pts = 'sum(participants_race)',
         groupby=['source', 'Year_Range', 'Race'],
         total='sum(NormalizedValueRace)',
         participants = 'sum(participants_race)',
         percentage = (('sum(participants_race)' / 'total_pts') *100)
     ).encode(
         theta=alt.Theta("total:Q", stack=True),
         color=alt.Color("Race:N", legend=None),
         tooltip=['source', 'Year_Range','Race', 'participants', 'percentage'],
     )

    # pie = base.mark_arc(outerRadius=80)

    # Define text labels

    #text = base.mark_text(radius=100, size=10).encode(text=alt.Text('Group:N') )


    #plot2 = (pie)

    #plot2



    # Add your funding-related charts here
     st.write("Charts related to Demographics theme")