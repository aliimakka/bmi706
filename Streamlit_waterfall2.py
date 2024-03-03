import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/Users/mattliebers/Downloads/bmi706/minus_OLE_with_generalized_indications_age_groups.csv')

# New multi-select sidebar option for seizure types
selected_seizure_types = st.sidebar.multiselect(
    'Select Seizure Types',
    options=['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status'],
    default=['Focal/Partial', 'Generalized', 'Epilepsy/Seizures/Status']
)

# Filter data based on selected seizure types
filtered_data_for_waterfall = data[data['indication_gen'].isin(selected_seizure_types)]

# Aggregate data by 'source' for the number of trials
sponsor_counts = filtered_data_for_waterfall['source'].value_counts()

# Separate the top 5 sponsors and group the rest as 'Other'
top_sponsors = sponsor_counts.head(5)
other_count = sponsor_counts[5:].sum()
final_counts = top_sponsors.append(pd.Series({'Other': other_count}))

# Preparing data for the waterfall plot
increments = final_counts.values
starts = np.zeros(len(increments))

# Calculating the start point for each bar
for i in range(1, len(increments)):
    starts[i] = starts[i-1] + increments[i-1]

# Plotting the waterfall plot
fig, ax = plt.subplots()
bars = plt.bar(final_counts.index, increments, bottom=starts)

# Adding labels and title
plt.title('Clinical Trials by Sponsor')
plt.xlabel('Sponsor')
plt.ylabel('Number of Trials')
plt.xticks(rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)
