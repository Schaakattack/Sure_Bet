# Import libraries and dependencies
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Load images
nfl_logo = Image.open('images/Logo.png')
players = Image.open('images/players.jpg')

# Read csv file with players data
df = pd.read_csv("data.csv")

# Group by player and get average scores
df_grouped = df.groupby("Player").mean().reset_index()

# Get back players' positions 
df = pd.merge(df_grouped, df[["Player","Position"]], on ='Player', how ='inner').drop_duplicates()

# Round scores to 2 decimlas values
df["FPTS"] = round(df["FPTS"],2)
df["FPTS/G"] = round(df["FPTS/G"],2)

# Load NFL Logo image in APP
st.image(nfl_logo, use_column_width="always")
#tryhing to add a gif image
#st.markdown("![Alt Text](https://giphy.com/gifs/nfl-49ers-san-francisco-l1AvAJ3Ixl96eBR60)")

# User's information input section on sidebar
st.sidebar.header("My Information")
st.sidebar.text_input("First Name")
st.sidebar.text_input("Last Name")
st.sidebar.text_input("Fantasy Team Name")
st.sidebar.radio("Gender",options=["Male","Female","Other"])
st.sidebar.number_input("Age",min_value=18, max_value=100, value=18, step=1)
st.sidebar.selectbox("Favorite Team", options=["Arizona Cardinals",
												"Atlanta Falcons",
												"Baltimore Ravens",
												"Buffalo Bills",
												"Carolina Panthers",
												"Chicago Bears",
												"Cincinnati Bengals",
												"Cleveland Browns",
												"Dallas Cowboys",
												"Denver Broncos",
												"Detroit Lions",
												"Green Bay Packers",
												"Houston Texans",
												"Indianapolis Colts",
												"Jacksonville Jaguars",
												"Kansas City Chiefs",
												"Miami Dolphins",
												"Minnesota Vikings",
												"New England Patriots",
												"New Orleans Saints",
												"NY Giants",
												"NY Jets",
												"Las Vegas Raiders",
												"Philadelphia Eagles",
												"Pittsburgh Steelers",
												"Los Angeles Chargers",
												"San Francisco 49ers",
												"Seattle Seahawks",
												"Los Angeles Rams",
												"Tampa Bay Buccaneers",
												"Tennessee Titans",
												"Washington Football Team"])

# Instructions section
st.header("Instructions")
st.markdown("Select players position. Then specifiy what round you will be drafting in. Click the 'Get best player available' button. The result will provide you with the best player available for your fantasy team.")
st.markdown("LETS GET STARTED")

# Drafting section
st.header("Place 'Sure' Bet!")

# Load players; image
st.image(players, use_column_width="always")

# Position radio button
position = st.radio("Please select the player's position:", options=df["Position"].unique())

# Round select 
round_ = st.number_input("Specify what round you are drafting in:",min_value=1, max_value=10, value=1, step=1)
# Run button
if st.button("Get best player available"):

	# Filter dataframe with selected position only
	df_sorted = df[df["Position"]==position].sort_values("FPTS", ascending=False)
	# Transpose dataframe
	st.dataframe(df_sorted.transpose().iloc[:,round_-1:round_])

	# K-Means Clustering Algorithm
	st.header("Players Clusters")
	st.markdown(f"Now that we have found the best player in the {position} position, it would be interesting to know how players could be grouped into 3 clusters: great players, decent players and bad players based on their historical Fanatsy Points.")
	st.markdown("In order to acheve this, lets apply **machine learning**! For this task, we will be applying an unsupervised machine learning technique for groupuing the players: K-Mean Clustering. Players in segment 2 will be considered as great players; players in segment 1 will be considered as decent players; and players in segment 0 will be considered as bad players.")

	# Set "Player" column as index for the dataframe
	df_sorted.set_index("Player",inplace=True)

	# Extract only FPTS and FPTs/G columns for analysis
	df_sorted = df_sorted[["FPTS","FPTS/G"]]

	# Instantiate KM model and fit it to the data
	model = KMeans(n_clusters=3, random_state=1).fit(df_sorted)
	
	# Get players segments
	player_segments = model.labels_
	
	# Add "Player Segment" column to dataframe
	df_sorted["Player Segment"] = player_segments

	# Reset the index
	df_sorted.reset_index(inplace=True)

	# Build interactive plotly scatter plot
	fig = px.scatter(
		df_sorted,x=df_sorted["FPTS"], 
		y=df_sorted["FPTS/G"], 
		color=df_sorted["Player Segment"], 
		hover_data=["Player"], 
		title="Player's Clusters based on Performance"
		
	)
	st.plotly_chart(fig)

	st.markdown("Great! The scatter plot above allows us to identify which player belongs to which cluster, and this should help us make a better decision when drafting them.")
	st.markdown(f"Finally, let's take a look at the data set of {position} players sorted by their Fantasy Points in descending order:")

	#3 Visualize filtered dataframe
	st.dataframe(df_sorted, width=5000)