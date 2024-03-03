import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Title for your Streamlit app
st.title('Seizure Type Comparison Across Age Groups')

# Load the dataset from the specified path
data = pd.read_csv('/Users/mattliebers/Downloads/bmi706/minus_OLE_with_generalized_indications_age_groups.csv')

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
st.pyplot(fig)




####################



# New multi-select sidebar option for seizure types
selected_seizure_types = st.sidebar.multiselect(
    'Select Seizure Types',
    options=['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status'],
    default=['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status']
)

# Filter data based on selected seizure types (if necessary for the waterfall plot)
filtered_data_for_waterfall = data[data['indication_gen'].isin(selected_seizure_types)]

# Aggregate data by 'source' for the number of trials
sponsor_counts = filtered_data_for_waterfall['source'].value_counts()

# Separate the top 5 sponsors and group the rest as 'Other'
top_sponsors = sponsor_counts.head(5)
other_count = sponsor_counts[5:].sum()
final_counts = top_sponsors.append(pd.Series({'Other': other_count}))

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

