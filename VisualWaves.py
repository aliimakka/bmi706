

import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import altair as alt
import numpy as np
from vega_datasets import data
import streamlit as st

#@st.cache

country_df = pd.read_csv('https://raw.githubusercontent.com/hms-dbmi/bmi706-2022/main/cancer_data/country_codes.csv', dtype={'country-code': str})
df = pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/country.csv')
df['totaltrials'] = df.groupby(['Study population', 'year', 'phase'])['Study population'].transform('count')
pharma = pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/pharma_country.csv', encoding='latin1')
pharma2=pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/minus_OLE_with_generalized_indications.csv')

merged_df = pd.merge(df, country_df[['Country', 'country-code']], left_on='Study population', right_on='Country', how='left').dropna()
merged_df = merged_df.dropna()
merged_df['year'] = merged_df['year'].astype(int)
merged_pharma = pd.merge(pharma, country_df[['Country', 'country-code']], left_on='Study population', right_on='Country', how='left')

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


race_df_ol=pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/age_ol.csv')
race_df_rct=pd.read_csv('https://raw.githubusercontent.com/aliimakka/bmi706/main/age_rct.csv')
combined_race_df = pd.concat([race_df_ol, race_df_rct], axis=0)


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
     st.pyplot(fig)


     # New multi-select sidebar option for seizure types
     selected_seizure_types = st.sidebar.multiselect(
        'Select Seizure Types',
      options=['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status'],
      default=['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status'])

     # Filter data based on selected seizure types
     filtered_data_for_waterfall = data[data['indication_gen'].isin(selected_seizure_types)]

     # Aggregate data by 'source' for the number of trials
     sponsor_counts = filtered_data_for_waterfall['source'].value_counts()

     # Separate the top 5 sponsors and group the rest as 'Other'
     top_sponsors = sponsor_counts.head(5)
     other_count = sponsor_counts[5:].sum()
     final_counts = pd.concat([top_sponsors, pd.Series({'Other': other_count})])


     # Preparing data for the waterfall plot
     increments = final_counts.values
     starts = np.zeros(len(increments))

     # Calculating the start point for each bar
     for i in range(1, len(increments)):
        starts[i] = starts[i-1] + increments[i-1]

     # Waterfall plot
     fig2 = go.Figure(go.Waterfall(
     name="20", orientation="v",
     measure=["relative"] * len(final_counts),
     x=final_counts.index,
     textposition="outside",
     text=final_counts.values,
     y=final_counts.values,
     connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))

     fig2.update_layout(title="Clinical Trials by Sponsor")

     # Display the plot in Streamlit
     st.plotly_chart(fig2)

      




elif selected_theme == "Demographics":



     
     st.subheader('Race breakdown by funding source over the years')

     df_race = pd.merge(df_filtered_by_phase[["ID", "year", 'source']], combined_race_df, on='ID', how='left').melt( 
     id_vars=["ID", "year",'source',],
     var_name="Race",
     value_name="participants",).drop_duplicates().groupby(['source', 'year','Race',]).agg({'participants': 'sum'}).reset_index()

     df_gender = df_filtered_by_phase[["ID", "year", 'source', 'phase', "Male", "Female"]].melt( 
     id_vars=["ID", "year", 'phase','source',],
     var_name="Gender",
     value_name="participants_gender",).drop_duplicates()
     charts = []

     unique_years_per_source = df_race.groupby('source')['year'].nunique()
     num_years = unique_years_per_source.max()

    
     if num_years > 11:
             year_bins = np.linspace(df_race['year'].min(), df_race['year'].max(), num=11)
             df_race['Year_Range'] = pd.cut(df_race['year'], bins=year_bins, include_lowest=True)
             df_race['Year_Range'] = df_race['Year_Range'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
     else:
             df_race['Year_Range'] = df_race['year'].astype(str)

     df_race['Proportion'] = (df_race.groupby(['source','Year_Range'])['participants'].transform(
                  lambda x: (x / x.sum())*100 if x.sum() != 0 else np.nan))
     
     df_filtered = df_race[df_race['Proportion'].notna()].sort_values(ascending=False, by='participants')


     source_selection_multi = alt.selection_multi(fields=['source'], bind='legend',on='click',empty="all")
     plotlin = alt.Chart(df_filtered).mark_bar().transform_aggregate(
             groupby=['source', 'Year_Range'],
             total='sum(participants)',
             ).encode(
                   x=alt.X('source:N', axis=None),
                   y='total:Q',
                   color=alt.Color('source:N',scale=alt.Scale(scheme='blueorange', reverse=True)),
                   column='Year_Range:O',
                   tooltip=['source','Year_Range', 'total:Q'],
                   ).add_selection(source_selection_multi).properties(
                title=f'Number of participants in trials sponsored by institutions from {selected_year[0]} to {selected_year[1]}'
           ).resolve_scale(x='independent')

     #df_filtered = df_filtered.dropna(subset=['source', 'Year_Range'])

     race_source_selection = alt.selection_single(fields=['source'], on='click',empty="all",clear='dblclick')
     for source in df_filtered['source'].unique():
          df_source = df_filtered[df_filtered['source'] == source]
          if df_source.empty: 
             continue
          pie = alt.Chart(df_source).mark_arc(outerRadius=40).transform_aggregate(
             groupby=['source', 'Year_Range', 'Race'],
             total='sum(participants)',
             proportion='mean(Proportion)'
             ).encode(
              theta=alt.Theta(f"proportion:Q", stack=True),
              color=alt.Color("Race:N",scale=alt.Scale(scheme='darkmulti'),  sort=['White', 'Black', 'Asian', 'Asian_Pacific Islander', 'other']),
              tooltip=['source', 'Year_Range','Race', 'total:Q'],
             ).transform_filter(source_selection_multi).properties(
              width=10,
              height=10).facet(
              column=alt.Column('Year_Range:N', header=alt.Header(title=None, labelColor='white')),
              title=f"{source}")
          charts.append(pie.add_selection(race_source_selection))
      

     final_chart = alt.vconcat(*charts).resolve_scale(x='independent', y='independent')


     plot3 = alt.Chart(df_filtered).mark_line(point=True).encode(
          x='year:N',
          y='participants:Q',
          color=alt.Color('Race:N', scale=alt.Scale(scheme='darkmulti')),
          tooltip=['source', 'Race:N', 'year', 'sum(participants)'],
          #).add_selection(race_source_selection
          ).transform_filter(source_selection_multi).properties(
              width=400,
              height=400,
                title=f'Race composition in trials sponsored by selected funding source from {selected_year[0]} to {selected_year[1]}'
           )

     
     plot_gender = alt.Chart(df_gender).mark_line(point=alt.OverlayMarkDef(filled=False, fill="white")).encode(
          x='year:O',
          y='sum(participants_gender):Q',
          shape='Gender:N',
          color=alt.Color('source:N', scale=alt.Scale(scheme='blueorange', reverse=True)),
          detail='Gender:N',
          tooltip=['source', 'Gender:N', 'year:Q', 'sum(participants_gender):Q'],
          #).add_selection(race_source_selection
         ).transform_filter(source_selection_multi).properties(
              width=400,
              height=400,
              title=f'Sex composition in trials sponsored by selected funding source from {selected_year[0]} to {selected_year[1]}'
           )
     
     plot_dem = (plot3 | plot_gender ).resolve_scale(color='independent', shape='independent')
     chart2= alt.vconcat(plotlin,final_chart,plot_dem).configure_legend(
            orient='right',
            padding=00,
            titleLimit=0,
            labelLimit=0
            ).resolve_scale(color='independent', shape='independent')
     

     
     st.altair_chart(chart2, use_container_width=True )




