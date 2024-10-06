import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load your dataset
df = pd.read_csv('spotify_cleaned.csv')

# Modify the artists column if not already modified
if 'artists_modified' not in df.columns:
    artists = []
    for row in df['artists']:
        item = row.split(',')
        artists.append(item)
    artists = [[word.strip() for word in inner_list] for inner_list in artists]
    df['artists'] = artists
    df['artists_modified'] = True

# Finding artists with the most songs in the dataset
artists_list = []
for item in df['artists']:
    artists_list.extend(item)
topArtists = pd.Series(artists_list).value_counts().head(10)

# Finding artists with the most streams
artists_data = {'artists': df['artists'], 'streams': df['streams']}
data = pd.DataFrame(artists_data)
df_exploded = data.explode('artists')

top_artists_df = df_exploded.groupby('artists')['streams'].sum().nlargest(10).reset_index()
df_artists = {'artists': topArtists.index, 'Count': topArtists.values}
top_tracks_df = df.groupby('track_name')['streams'].sum().nlargest(10).reset_index()

# Streamlit code
st.title('🎵 Music Data Analysis Dashboard')

# Sidebar for navigation (Top 10 Visualizations)
st.sidebar.header('Navigation')
option = st.sidebar.radio('Select Plot Type:', 
                          ['Top 10 Artists', 'Top 10 Artists With Most Song Count', 'Top 10 Tracks'])

# ---- Panel 1: Top 10 Visualizations based on sidebar option ----
st.subheader('Top 10 Visualizations')

if option == 'Top 10 Artists':
    data, title, xlabel, ylabel = top_artists_df, 'Top 10 Artists Based on Streams', 'streams', 'artists'
elif option == 'Top 10 Tracks':
    data, title, xlabel, ylabel = top_tracks_df, 'Top 10 Tracks Based on Streams', 'streams', 'track_name'
elif option == 'Top 10 Artists With Most Song Count':
    data, title, xlabel, ylabel = df_artists, 'Top 10 Artists with Most Songs', 'Count', 'artists'

# Set dark mode or transparent background for plots
plt.style.use('dark_background')  # Comment out if you prefer transparent background

# Plot the selected visualization
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=xlabel, y=ylabel, data=data, palette='magma', ax=ax)
ax.set_xlabel(xlabel.title())
ax.set_ylabel(ylabel.replace('_', ' ').title())
ax.set_title(title, fontsize=14, weight='bold')

# Make background transparent to match Streamlit's dark/light theme
fig.patch.set_facecolor('None')  # Set to transparent if dark_background not needed
st.pyplot(fig)

# ---- Panel 2: Audio Features Analysis ----
st.subheader('🎶 Audio Features Analysis')

# Tabs for different visualizations inside the Audio Features Analysis section
audio_tab1, audio_tab2, audio_tab3 = st.tabs(['Scatter Plot', 'Bar Plot', 'Correlation Matrix'])

with audio_tab1:
    st.subheader('Audio Features vs Number of Streams')
    st.write("Explore how audio features like danceability, energy, and valence relate to the number of streams.")
    
    feature_columns = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm']
    feature = st.selectbox('Select Feature:', feature_columns)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=df[feature], y=df['streams'], hue=df[feature], data=df, palette='plasma', ax=ax)
    ax.set_title(f'{feature.replace("_%", "").title()} vs Number of Streams', fontsize=12)
    ax.set_xlabel(feature.replace('_%', '').title())
    ax.set_ylabel('Number of Streams (in billions)')
    fig.patch.set_facecolor('none')  # Set to transparent to match background
    st.pyplot(fig)

with audio_tab2:
    st.subheader('Barplot of Streams by Key and Mode')
    st.write("This plot shows the distribution of streams based on the musical key and mode of the songs.")
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=df['key'], y=df['streams'], hue=df['mode'], data=df, ci=None, ax=ax)
    ax.set_title('Barplot of Streams by Key and Mode', fontsize=12)
    ax.set_xlabel('Key')
    ax.set_ylabel('Number of Streams (in billions)')
    fig.patch.set_facecolor('none')  # Set to transparent to match background
    st.pyplot(fig)

with audio_tab3:
    st.subheader('Correlation Matrix for Audio Features')
    st.write("Check the correlations between different audio features to see how they are related.")
    
    corr_matrix = df[feature_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='mako', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix for Audio Features', fontsize=12)
    fig.patch.set_facecolor('none')  # Set to transparent to match background
    st.pyplot(fig)
