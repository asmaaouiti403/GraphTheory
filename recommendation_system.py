import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import time


# Constants
TMDB_API_KEY = "bb837076a7975b3d62f18e59b0dfa643a"
TMDB_BASE_URL = "https://api.themoviedb.org/3"


# Set page configuration
st.set_page_config(
   page_title="Movie Graph Explorer",
   page_icon="🎬",
   layout="wide"
)


# Cache functions to improve performance
@st.cache_data(ttl=3600)
def fetch_popular_movies(page=1):
   """Fetch popular movies from TMDB API"""
   url = f"{TMDB_BASE_URL}/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}"
   response = requests.get(url)
   if response.status_code == 200:
       return response.json()['results']
   else:
       st.error(f"Error fetching popular movies: {response.status_code}")
       return []


@st.cache_data(ttl=3600)
def fetch_movie_details(movie_id):
   """Fetch detailed movie information including genres and similar movies"""
   url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=similar,credits"
   response = requests.get(url)
   if response.status_code == 200:
       return response.json()
   else:
       st.error(f"Error fetching movie details: {response.status_code}")
       return None


@st.cache_data(ttl=3600)
def search_movies(query):
   """Search for movies by title"""
   url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&language=en-US&query={query}&page=1"
   response = requests.get(url)
   if response.status_code == 200:
       return response.json()['results']
   else:
       st.error(f"Error searching movies: {response.status_code}")
       return []


@st.cache_data(ttl=3600)
def fetch_genre_list():
   """Fetch list of movie genres"""
   url = f"{TMDB_BASE_URL}/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
   response = requests.get(url)
   if response.status_code == 200:
       return {genre['id']: genre['name'] for genre in response.json()['genres']}
   else:
       st.error(f"Error fetching genres: {response.status_code}")
       return {}


@st.cache_data(ttl=3600)
def discover_movies_by_genre(genre_id, page=1):
   """Discover movies by genre"""
   url = f"{TMDB_BASE_URL}/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&page={page}"
   response = requests.get(url)
   if response.status_code == 200:
       return response.json()['results']
   else:
       st.error(f"Error discovering movies: {response.status_code}")
       return []


# Initialize session state for user data
if 'rated_movies' not in st.session_state:
   st.session_state.rated_movies = {}  # {movie_id: movie_data}

if 'ratings' not in st.session_state:
   st.session_state.ratings = {}  # {movie_id: rating}

if 'rate_this_movie' not in st.session_state:
   st.session_state.rate_this_movie = None


# Functions for graph operations
def create_movie_similarity_graph(movie_id, similar_movies, threshold=0.5):
   """Create a graph showing movie similarity relationships"""
   G = nx.Graph()
  
   # Get main movie details
   main_movie = fetch_movie_details(movie_id)
   if not main_movie:
       return None
  
   main_title = main_movie['title']
   G.add_node(main_title, type='main')
  
   # Add similar movies as nodes
   for movie in similar_movies[:10]:  # Limit to top 10 similar movies
       similarity = movie.get('vote_average', 5) / 10  # Normalize to 0-1
       if similarity >= threshold:
           G.add_node(movie['title'], type='similar')
           G.add_edge(main_title, movie['title'], weight=similarity)
  
   return G


def create_genre_graph(movies_data):
   """Create a graph showing relationships between genres based on movies"""
   G = nx.Graph()
   genre_dict = fetch_genre_list()
   genre_connections = {}
  
   # Count genre co-occurrences
   for movie in movies_data:
       if 'genre_ids' in movie and len(movie['genre_ids']) > 1:
           for i, genre_id1 in enumerate(movie['genre_ids']):
               for genre_id2 in movie['genre_ids'][i+1:]:
                   if genre_id1 in genre_dict and genre_id2 in genre_dict:
                       pair = tuple(sorted([genre_dict[genre_id1], genre_dict[genre_id2]]))
                       genre_connections[pair] = genre_connections.get(pair, 0) + 1
  
   # Add nodes and edges to graph
   for genre_id, genre_name in genre_dict.items():
       G.add_node(genre_name)
  
   for (genre1, genre2), weight in genre_connections.items():
       G.add_edge(genre1, genre2, weight=weight)
  
   return G


def create_rating_quality_graph(rating_category):
   """Create a graph showing movies with specified rating quality"""
   G = nx.Graph()
   
   # Define rating categories
   category_labels = {
       "low": "Low Rated Movies (1-2 ⭐)",
       "average": "Average Rated Movies (3 ⭐)",
       "high": "High Rated Movies (4-5 ⭐)"
   }
   
   # Add central node
   center_node = category_labels[rating_category]
   G.add_node(center_node, type='category', size=800)
   
   # Filter movies by rating category
   movie_nodes = []
   for movie_id, rating in st.session_state.ratings.items():
       if movie_id in st.session_state.rated_movies:
           movie = st.session_state.rated_movies[movie_id]
           if (rating_category == "low" and rating <= 2) or \
              (rating_category == "average" and rating == 3) or \
              (rating_category == "high" and rating >= 4):
               movie_title = movie['title']
               # Add movie node
               G.add_node(movie_title, type='movie', rating=rating, size=250)
               movie_nodes.append(movie_title)
               # Connect to category center
               G.add_edge(movie_title, center_node, weight=1)
   
   # Connect movies with similar genres
   movie_genres = {}
   for movie_id, rating in st.session_state.ratings.items():
       if movie_id in st.session_state.rated_movies:
           movie = st.session_state.rated_movies[movie_id]
           # Only include movies in the current rating category
           if (rating_category == "low" and rating <= 2) or \
              (rating_category == "average" and rating == 3) or \
              (rating_category == "high" and rating >= 4):
               if 'genre_ids' in movie:
                   movie_genres[movie['title']] = set(movie.get('genre_ids', []))
   
   # Connect movies by common genres
   for title1, genres1 in movie_genres.items():
       for title2, genres2 in movie_genres.items():
           if title1 != title2:
               common_genres = genres1.intersection(genres2)
               if common_genres:
                   G.add_edge(title1, title2, weight=len(common_genres)/2)
   
   return G


def visualize_graph(G, title="Movie Graph", rating_category=None):
   """Visualize a NetworkX graph"""
   plt.figure(figsize=(10, 8))
   plt.title(title)
  
   # Customize layout and appearance
   if 'type' in G.nodes[list(G.nodes)[0]]:
       # Check if it's a rating quality graph
       if any(G.nodes[node].get('type') == 'category' for node in G.nodes):
           pos = nx.spring_layout(G, seed=42, k=0.3)
           
           # Filter nodes by type
           category_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'category']
           movie_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'movie']
           
           # Choose color based on rating category
           if rating_category == "low":
               category_color = '#F44336'  # Red for low ratings
               movie_color = '#FFCDD2'     # Light red
           elif rating_category == "average":
               category_color = '#FFC107'  # Yellow for average ratings
               movie_color = '#FFF9C4'     # Light yellow
           else:  # high
               category_color = '#4CAF50'  # Green for high ratings
               movie_color = '#A5D6A7'     # Light green
           
           # Draw category nodes
           nx.draw_networkx_nodes(G, pos, 
                                 nodelist=category_nodes,
                                 node_color=[category_color],
                                 node_size=[G.nodes[node].get('size', 800) for node in category_nodes],
                                 alpha=0.8)
           
           # Draw movie nodes
           if movie_nodes:
               nx.draw_networkx_nodes(G, pos,
                                      nodelist=movie_nodes,
                                      node_color=movie_color,
                                      node_size=[G.nodes[node].get('size', 250) for node in movie_nodes],
                                      alpha=0.7)
           
           # Draw edges
           nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray')
           
           # Draw labels with different sizes
           category_labels = {node: node for node in category_nodes}
           movie_labels = {node: node for node in movie_nodes}
           
           nx.draw_networkx_labels(G, pos, labels=category_labels, font_size=14, font_weight='bold')
           nx.draw_networkx_labels(G, pos, labels=movie_labels, font_size=8)
       
       # Movie similarity graph
       elif any(G.nodes[node].get('type') == 'main' for node in G.nodes):
           pos = nx.spring_layout(G, seed=42)
           node_colors = ['red' if G.nodes[node].get('type') == 'main' else 'skyblue' for node in G.nodes]
           node_sizes = [800 if G.nodes[node].get('type') == 'main' else 300 for node in G.nodes]
          
           # Draw graph elements
           nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
          
           # Edge weights represent similarity
           edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges]
           nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
           nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
      
   else:
       # Genre relationship graph
       pos = nx.spring_layout(G, seed=42, k=0.5)
      
       # Size nodes based on degree (popularity)
       node_sizes = [100 + G.degree(node) * 50 for node in G.nodes]
      
       # Draw graph elements with edge width based on weight
       nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=node_sizes, alpha=0.8)
      
       edge_weights = [G[u][v]['weight'] / 5 for u, v in G.edges]
       nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
       nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
  
   plt.axis('off')
   return plt


# User Interface
def main():
   st.title("🎬 Movie Graph Explorer")
   st.sidebar.title("Navigation")
  
   # Navigation sidebar
   pages = [
       "🎲 Discover New Movies",
       "🔍 Find Similar Movies",
       "🎭 Genre Explorer",
       "📊 My Movie Ratings"
   ]
  
   selection = st.sidebar.radio("Go to", pages)
   
   # Add a movie search and rating section in the sidebar
   with st.sidebar:
       st.markdown("---")
       st.subheader("Rate a Movie")
       sidebar_movie_query = st.text_input("Search for a movie to rate:", key="sidebar_movie_search")
       
       if sidebar_movie_query:
           search_results = search_movies(sidebar_movie_query)
           if search_results:
               for i, movie in enumerate(search_results[:3]):  # Limit to 3 results
                   cols = st.columns([1, 2])
                   with cols[0]:
                       if movie.get('poster_path'):
                           st.image(f"https://image.tmdb.org/t/p/w92{movie['poster_path']}")
                   with cols[1]:
                       st.write(f"**{movie['title']}**")
                       # Store movie in rated_movies if not already present
                       if movie['id'] not in st.session_state.rated_movies:
                           st.session_state.rated_movies[movie['id']] = movie
                       
                       # Show current rating if it exists
                       current_rating = st.session_state.ratings.get(movie['id'], 3)
                       rating = st.slider(
                           "Rating", 
                           1, 5, current_rating, 
                           key=f"sidebar_rating_{movie['id']}"
                       )
                       
                       if st.button("Submit Rating", key=f"sidebar_submit_{movie['id']}"):
                           st.session_state.ratings[movie['id']] = rating
                           st.success(f"Rating for '{movie['title']}' submitted!")
                           st.experimental_rerun()
           else:
               st.info("No movies found with that title.")
  
   # Fetch general data
   genres = fetch_genre_list()
  
   # Page specific content
   if selection == "🎲 Discover New Movies":
       discover_new_movies(genres)
  
   elif selection == "🔍 Find Similar Movies":
       find_similar_movies()
  
   elif selection == "🎭 Genre Explorer":
       genre_explorer(genres)
  
   elif selection == "📊 My Movie Ratings":
       rating_visualization()
  
   # Footer
   st.sidebar.markdown("---")
   st.sidebar.info(
       """
       This app uses graph theory concepts to explore movie relationships.
      
       - Nodes: Movies, genres, and rating categories
       - Edges: Similarities and relationships
       - Rating groups: Good (4-5 ⭐), Average (3 ⭐), Bad (1-2 ⭐)
       """
   )


def discover_new_movies(genres):
   st.header("Discover New Movies")
  
   # Genre filter
   genre_options = list(genres.values())
   genre_options.insert(0, "All Genres")
   selected_genre = st.selectbox("Select a genre", genre_options)
  
   # Get genre ID
   genre_id = None
   if selected_genre != "All Genres":
       genre_id = [k for k, v in genres.items() if v == selected_genre][0]
  
   st.subheader("Random Movie Suggestions")
  
   if st.button("Get Random Recommendations"):
       with st.spinner("Fetching recommendations..."):
           if genre_id:
               # Fetch movies from specific genre
               page = random.randint(1, 5)
               movies = discover_movies_by_genre(genre_id, page)
           else:
               # Fetch popular movies from random page
               page = random.randint(1, 10)
               movies = fetch_popular_movies(page)
          
           # Randomly choose 4 movies
           if movies:
               random_movies = random.sample(movies, min(4, len(movies)))
              
               # Display in a grid
               cols = st.columns(2)
               for i, movie in enumerate(random_movies):
                   col_idx = i % 2
                   with cols[col_idx]:
                       st.subheader(movie['title'])
                       if movie.get('poster_path'):
                           st.image(f"https://image.tmdb.org/t/p/w300{movie['poster_path']}")
                       st.write(f"**Rating:** {movie.get('vote_average', 'N/A')}/10")
                       st.write(f"**Release Date:** {movie.get('release_date', 'Unknown')}")
                      
                       # Show overview with a "Read more" expander if it's long
                       overview = movie.get('overview', 'No description available.')
                       if len(overview) > 150:
                           st.write(f"{overview[:150]}...")
                           with st.expander("Read more"):
                               st.write(overview)
                       else:
                           st.write(overview)


def find_similar_movies():
   st.header("Find Similar Movies")
  
   # Input for movie title
   movie_query = st.text_input("Enter a movie title:", key="movie_search")
  
   if movie_query:
       with st.spinner("Searching..."):
           search_results = search_movies(movie_query)
          
           if search_results:
               st.subheader("Select a Movie")
              
               for i, movie in enumerate(search_results[:5]):  # Limit to 5 results
                   cols = st.columns([1, 3])
                   with cols[0]:
                       if movie.get('poster_path'):
                           st.image(f"https://image.tmdb.org/t/p/w92{movie['poster_path']}")
                   with cols[1]:
                       st.write(f"**{movie['title']}** ({movie.get('release_date', 'Unknown')[:4] if movie.get('release_date') else 'Unknown'})")
                      
                       if st.button("Show Similar Movies", key=f"similar_{movie['id']}"):
                           with st.spinner("Building movie graph..."):
                               # Store movie in rated_movies but don't prompt for rating
                               st.session_state.rated_movies[movie['id']] = movie
                               
                               # Fetch detailed movie info
                               movie_details = fetch_movie_details(movie['id'])
                              
                               if movie_details and 'similar' in movie_details:
                                   similar_movies = movie_details['similar']['results']
                                  
                                   # Create and visualize graph
                                   G = create_movie_similarity_graph(movie['id'], similar_movies)
                                   if G and len(G.nodes) > 1:
                                       st.subheader(f"Movies Similar to '{movie['title']}'")
                                      
                                       # Visualization
                                       plt = visualize_graph(G, title=f"Movies Similar to '{movie['title']}'")
                                       st.pyplot(plt)
                                      
                                       # List similar movies
                                       st.subheader("Recommended Movies")
                                       movie_cols = st.columns(2)
                                       for idx, sim_movie in enumerate(similar_movies[:6]):
                                           col_idx = idx % 2
                                           with movie_cols[col_idx]:
                                               st.write(f"**{sim_movie['title']}**")
                                               if sim_movie.get('poster_path'):
                                                   st.image(f"https://image.tmdb.org/t/p/w154{sim_movie['poster_path']}")
                                               st.write(f"Similarity: {sim_movie.get('vote_average')/10:.2f}")
                                               # Store in rated_movies but don't show rating UI
                                               st.session_state.rated_movies[sim_movie['id']] = sim_movie
                                   else:
                                       st.warning("Not enough similar movies found to create a graph.")
                               else:
                                   st.error("Couldn't fetch similar movies.")
           else:
               st.warning("No movies found with that title. Please try another search.")


def genre_explorer(genres):
   st.header("Genre Explorer")
  
   # Multi-select for genres
   selected_genres = st.multiselect("Select genres to explore", list(genres.values()), default=[list(genres.values())[0]])
  
   if selected_genres:
       # Get movies for selected genres
       movies_data = []
       with st.spinner("Fetching movies and building genre graph..."):
           for genre_name in selected_genres:
               genre_id = [k for k, v in genres.items() if v == genre_name][0]
               # Get movies from multiple pages for better data
               for page in range(1, 3):
                   genre_movies = discover_movies_by_genre(genre_id, page)
                   movies_data.extend(genre_movies)
          
           if movies_data:
               # Create and visualize genre graph
               G = create_genre_graph(movies_data)
              
               if G and len(G.edges) > 0:
                   st.subheader("Genre Relationship Graph")
                   st.write("This graph shows how different genres are connected through movies. Thicker edges indicate stronger connections.")
                  
                   # Visualization
                   plt = visualize_graph(G, title="Genre Relationships")
                   st.pyplot(plt)
                  
                   # Stats about the graph
                   st.subheader("Graph Statistics")
                   col1, col2, col3 = st.columns(3)
                   with col1:
                       st.metric("Genres", len(G.nodes))
                   with col2:
                       st.metric("Connections", len(G.edges))
                   with col3:
                       degrees = dict(G.degree())
                       most_connected = max(degrees.items(), key=lambda x: x[1])
                       st.metric("Most Connected Genre", f"{most_connected[0]} ({most_connected[1]})")
                  
                   # Show movies that have multiple selected genres
                   multi_genre_movies = []
                   for movie in movies_data:
                       movie_genres = [genres.get(genre_id) for genre_id in movie.get('genre_ids', [])]
                       common_genres = set(movie_genres) & set(selected_genres)
                       if len(common_genres) >= 2:
                           movie['common_genres'] = list(common_genres)
                           multi_genre_movies.append(movie)
                  
                   if multi_genre_movies:
                       st.subheader(f"Movies with Multiple Selected Genres ({len(multi_genre_movies)})")
                      
                       # Pagination for multi-genre movies
                       items_per_page = 4
                       pages = len(multi_genre_movies) // items_per_page + (1 if len(multi_genre_movies) % items_per_page > 0 else 0)
                      
                       # Page selector
                       if pages > 1:
                           page_num = st.selectbox("Page", range(1, pages + 1))
                       else:
                           page_num = 1
                      
                       # Display movies for current page
                       start_idx = (page_num - 1) * items_per_page
                       end_idx = min(start_idx + items_per_page, len(multi_genre_movies))
                      
                       for movie in multi_genre_movies[start_idx:end_idx]:
                           cols = st.columns([1, 3])
                           with cols[0]:
                               if movie.get('poster_path'):
                                   st.image(f"https://image.tmdb.org/t/p/w154{movie['poster_path']}")
                           with cols[1]:
                               st.write(f"**{movie['title']}**")
                               st.write(f"Genres: {', '.join(movie['common_genres'])}")
                               # Store in rated_movies but don't show rating UI
                               st.session_state.rated_movies[movie['id']] = movie
               else:
                   st.warning("Not enough genre connections found to create a graph.")
           else:
               st.error("Couldn't fetch movies for the selected genres.")


def rating_visualization():
   st.header("My Movie Ratings")
  
   if not st.session_state.ratings:
       st.info("You haven't rated any movies yet. Rate some movies in the sidebar to see your personalized graphs.")
   else:
       # Create three separate tabs for each rating category graph
       st.subheader("Rating Graphs by Category")
       
       # Get counts by category
       low_count = sum(1 for r in st.session_state.ratings.values() if r <= 2)
       avg_count = sum(1 for r in st.session_state.ratings.values() if r == 3)
       high_count = sum(1 for r in st.session_state.ratings.values() if r >= 4)
       
       # Create tabs for each graph
       tabs = st.tabs([
           f"Low Rated Movies ({low_count})",
           f"Average Rated Movies ({avg_count})",
           f"High Rated Movies ({high_count})"
       ])
       
       # Low rated movies graph
       with tabs[0]:
           if low_count > 0:
               st.write("Graph showing movies you rated 1-2 ⭐")
               G_low = create_rating_quality_graph("low")
               if len(G_low.nodes) > 1:  # Check if we have more than just the category node
                   plt = visualize_graph(G_low, title="Low Rated Movies", rating_category="low")
                   st.pyplot(plt)
                   
                   # Display list of low rated movies
                   st.subheader("Movies in this Category")
                   display_rated_movie_list("low")
               else:
                   st.info("You haven't rated any movies in this category yet.")
           else:
               st.info("You haven't rated any movies in this category yet.")
       
       # Average rated movies graph
       with tabs[1]:
           if avg_count > 0:
               st.write("Graph showing movies you rated 3 ⭐")
               G_avg = create_rating_quality_graph("average")
               if len(G_avg.nodes) > 1:
                   plt = visualize_graph(G_avg, title="Average Rated Movies", rating_category="average")
                   st.pyplot(plt)
                   
                   # Display list of average rated movies
                   st.subheader("Movies in this Category")
                   display_rated_movie_list("average")
               else:
                   st.info("You haven't rated any movies in this category yet.")
           else:
               st.info("You haven't rated any movies in this category yet.")
       
       # High rated movies graph
       with tabs[2]:
           if high_count > 0:
               st.write("Graph showing movies you rated 4-5 ⭐")
               G_high = create_rating_quality_graph("high")
               if len(G_high.nodes) > 1:
                   plt = visualize_graph(G_high, title="High Rated Movies", rating_category="high")
                   st.pyplot(plt)
                   
                   # Display list of high rated movies
                   st.subheader("Movies in this Category")
                   display_rated_movie_list("high")
               else:
                   st.info("You haven't rated any movies in this category yet.")
           else:
               st.info("You haven't rated any movies in this category yet.")

       # Add a section for rating statistics
       st.subheader("Rating Statistics")
       col1, col2, col3 = st.columns(3)
       with col1:
           st.metric("Good Movies (4-5 ⭐)", high_count)
       with col2:
           st.metric("Average Movies (3 ⭐)", avg_count)
       with col3:
           st.metric("Bad Movies (1-2 ⭐)", low_count)
       
       # Genre preference analysis if there are enough ratings
       if high_count > 0:
           st.subheader("Genre Preferences Analysis")
           analyze_genre_preferences()


def display_rated_movie_list(rating_category):
   """Display a list of movies in the specified rating category"""
   # Filter movies by rating category
   movies = []
   for movie_id, rating in st.session_state.ratings.items():
       if movie_id in st.session_state.rated_movies:
           movie = st.session_state.rated_movies[movie_id]
           if (rating_category == "low" and rating <= 2) or \
              (rating_category == "average" and rating == 3) or \
              (rating_category == "high" and rating >= 4):
               movies.append({**movie, 'user_rating': rating})
   
   # Sort by title
   movies.sort(key=lambda x: x['title'])
   
   # Display in a grid
   cols = st.columns(3)
   for i, movie in enumerate(movies):
       col_idx = i % 3
       with cols[col_idx]:
           if movie.get('poster_path'):
               st.image(f"https://image.tmdb.org/t/p/w154{movie['poster_path']}")
           st.write(f"**{movie['title']}**")
           st.write(f"Your Rating: {'⭐' * movie['user_rating']}")
           
           # Display genres if available
           if 'genre_ids' in movie:
               genre_dict = fetch_genre_list()
               genres = [genre_dict.get(gid, "Unknown") for gid in movie['genre_ids'] if gid in genre_dict]
               if genres:
                   st.write(f"Genres: {', '.join(genres)}")


def analyze_genre_preferences():
   """Analyze user's genre preferences based on ratings"""
   good_movie_ids = [movie_id for movie_id, rating in st.session_state.ratings.items() if rating >= 4]
   bad_movie_ids = [movie_id for movie_id, rating in st.session_state.ratings.items() if rating <= 2]
   
   good_movie_genres = []
   bad_movie_genres = []
   
   genre_dict = fetch_genre_list()
   
  # Collect genres for good and bad movies
   genre_dict = fetch_genre_list()
   
   # Collect genres for good and bad rated movies
   for movie_id in good_movie_ids:
       if movie_id in st.session_state.rated_movies:
           movie = st.session_state.rated_movies[movie_id]
           if 'genre_ids' in movie:
               good_movie_genres.extend([genre_dict.get(gid) for gid in movie['genre_ids'] if gid in genre_dict])
   
   for movie_id in bad_movie_ids:
       if movie_id in st.session_state.rated_movies:
           movie = st.session_state.rated_movies[movie_id]
           if 'genre_ids' in movie:
               bad_movie_genres.extend([genre_dict.get(gid) for gid in movie['genre_ids'] if gid in genre_dict])
   
   # Count genre occurrences
   from collections import Counter
   good_genre_counts = Counter(good_movie_genres)
   bad_genre_counts = Counter(bad_movie_genres)
   
   # Display results as a bar chart if there are enough ratings
   if good_genre_counts or bad_genre_counts:
       # Prepare data for visualization
       all_genres = set(list(good_genre_counts.keys()) + list(bad_genre_counts.keys()))
       chart_data = []
       
       for genre in all_genres:
           chart_data.append({
               'Genre': genre,
               'Liked (4-5 ⭐)': good_genre_counts.get(genre, 0),
               'Disliked (1-2 ⭐)': bad_genre_counts.get(genre, 0)
           })
       
       # Convert to DataFrame for easier visualization
       chart_df = pd.DataFrame(chart_data)
       
       # Sort by highest positive rating difference
       chart_df['Difference'] = chart_df['Liked (4-5 ⭐)'] - chart_df['Disliked (1-2 ⭐)']
       chart_df = chart_df.sort_values('Difference', ascending=False)
       
       # Create two columns for visualization
       col1, col2 = st.columns([3, 2])
       
       with col1:
           # Create a bar chart
           st.subheader("Genre Preference Distribution")
           
           chart_df_display = chart_df.drop(columns=['Difference'])
           st.bar_chart(chart_df_display.set_index('Genre'))
       
       with col2:
           # List favorite and least favorite genres
           st.subheader("Your Genre Preferences")
           
           # Favorite genres (most positive difference)
           if not chart_df.empty and chart_df['Difference'].max() > 0:
               favorite_genres = chart_df[chart_df['Difference'] > 0].sort_values('Difference', ascending=False)
               if not favorite_genres.empty:
                   st.write("**Favorite Genres:**")
                   for _, row in favorite_genres.head(3).iterrows():
                       st.write(f"- {row['Genre']} (+{row['Difference']})")
           
           # Least favorite genres (most negative difference)
           if not chart_df.empty and chart_df['Difference'].min() < 0:
               least_favorite = chart_df[chart_df['Difference'] < 0].sort_values('Difference', ascending=True)
               if not least_favorite.empty:
                   st.write("**Least Favorite Genres:**")
                   for _, row in least_favorite.head(3).iterrows():
                       st.write(f"- {row['Genre']} ({row['Difference']})")
   else:
       st.info("Rate more movies to see your genre preferences analysis.")


def get_recommendation_based_on_ratings():
   """Generate movie recommendations based on user ratings"""
   # Check if we have enough ratings
   if len(st.session_state.ratings) < 3:
       return None
   
   # Get highly rated movies
   highly_rated = [movie_id for movie_id, rating in st.session_state.ratings.items() if rating >= 4]
   
   if not highly_rated:
       return None
   
   # Find genres of highly rated movies
   genre_dict = fetch_genre_list()
   favorite_genres = []
   
   for movie_id in highly_rated:
       if movie_id in st.session_state.rated_movies:
           movie = st.session_state.rated_movies[movie_id]
           if 'genre_ids' in movie:
               favorite_genres.extend([genre_id for genre_id in movie['genre_ids'] if genre_id in genre_dict])
   
   # Count and find most common genres
   from collections import Counter
   genre_counts = Counter(favorite_genres)
   
   if not genre_counts:
       return None
   
   # Get top 2 genres
   top_genres = [genre_id for genre_id, _ in genre_counts.most_common(2)]
   
   if not top_genres:
       return None
   
   # Get movies from these genres
   genre_string = ",".join(str(g) for g in top_genres)
   recommendations = []
   
   # Try to get from multiple pages for better variety
   for page in range(1, 3):
       url = f"{TMDB_BASE_URL}/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_string}&page={page}&sort_by=popularity.desc"
       response = requests.get(url)
       
       if response.status_code == 200:
           results = response.json().get('results', [])
           # Filter out movies already rated
           new_movies = [movie for movie in results if movie['id'] not in st.session_state.ratings]
           recommendations.extend(new_movies)
   
   # Return top recommendations
   return recommendations[:6] if recommendations else None

def rating_visualization():
   st.header("My Movie Ratings")
  
   if not st.session_state.ratings:
       st.info("You haven't rated any movies yet. Rate some movies below or in the sidebar to see your personalized graphs.")
       # Add a search box to find and rate movies directly on this page
       st.subheader("Rate Movies")
       main_movie_query = st.text_input("Search for a movie to rate:", key="main_movie_search")
       
       if main_movie_query:
           search_results = search_movies(main_movie_query)
           if search_results:
               cols = st.columns(3)
               for i, movie in enumerate(search_results[:6]):  # Show up to 6 results
                   col_idx = i % 3
                   with cols[col_idx]:
                       st.write(f"**{movie['title']}**")
                       if movie.get('poster_path'):
                           st.image(f"https://image.tmdb.org/t/p/w154{movie['poster_path']}")
                       
                       # Store movie in rated_movies if not already present
                       if movie['id'] not in st.session_state.rated_movies:
                           st.session_state.rated_movies[movie['id']] = movie
                       
                       # Show current rating if it exists
                       current_rating = st.session_state.ratings.get(movie['id'], 3)
                       rating = st.slider(
                           "Rating", 
                           1, 5, current_rating, 
                           key=f"main_rating_{movie['id']}"
                       )
                       
                       if st.button("Submit Rating", key=f"main_submit_{movie['id']}"):
                           st.session_state.ratings[movie['id']] = rating
                           st.success(f"Rating for '{movie['title']}' submitted!")
                           st.experimental_rerun()
           else:
               st.info("No movies found with that title.")
   else:
       # Option to view by tabs or combined graph
       view_option = st.radio(
           "View Options:", 
           ["Combined Graph", "Separate Tabs"], 
           horizontal=True
       )
       
       # Get counts by category
       low_count = sum(1 for r in st.session_state.ratings.values() if r <= 2)
       avg_count = sum(1 for r in st.session_state.ratings.values() if r == 3)
       high_count = sum(1 for r in st.session_state.ratings.values() if r >= 4)
       
       if view_option == "Combined Graph":
           st.subheader("All Movie Ratings Visualization")
           
           # Create combined graph showing all rated movies
           G_combined = create_combined_rating_graph()
           if len(G_combined.nodes) > 3:  # Ensure we have enough nodes besides the category centers
               plt = visualize_combined_graph(G_combined)
               st.pyplot(plt)
               
               # Display list of all rated movies
               st.subheader("All Your Rated Movies")
               display_all_rated_movies()
           else:
               st.info("Rate more movies to see a meaningful graph.")
       else:
           # Create tabs for each graph (your existing code)
           tabs = st.tabs([
               f"Low Rated Movies ({low_count})",
               f"Average Rated Movies ({avg_count})",
               f"High Rated Movies ({high_count})"
           ])
           
           # Low rated movies graph
           with tabs[0]:
               if low_count > 0:
                   st.write("Graph showing movies you rated 1-2 ⭐")
                   G_low = create_rating_quality_graph("low")
                   if len(G_low.nodes) > 1:  # Check if we have more than just the category node
                       plt = visualize_graph(G_low, title="Low Rated Movies", rating_category="low")
                       st.pyplot(plt)
                       
                       # Display list of low rated movies
                       st.subheader("Movies in this Category")
                       display_rated_movie_list("low")
                   else:
                       st.info("You haven't rated any movies in this category yet.")
               else:
                   st.info("You haven't rated any movies in this category yet.")
           
           # Average rated movies graph
           with tabs[1]:
               if avg_count > 0:
                   st.write("Graph showing movies you rated 3 ⭐")
                   G_avg = create_rating_quality_graph("average")
                   if len(G_avg.nodes) > 1:
                       plt = visualize_graph(G_avg, title="Average Rated Movies", rating_category="average")
                       st.pyplot(plt)
                       
                       # Display list of average rated movies
                       st.subheader("Movies in this Category")
                       display_rated_movie_list("average")
                   else:
                       st.info("You haven't rated any movies in this category yet.")
               else:
                   st.info("You haven't rated any movies in this category yet.")
           
           # High rated movies graph
           with tabs[2]:
               if high_count > 0:
                   st.write("Graph showing movies you rated 4-5 ⭐")
                   G_high = create_rating_quality_graph("high")
                   if len(G_high.nodes) > 1:
                       plt = visualize_graph(G_high, title="High Rated Movies", rating_category="high")
                       st.pyplot(plt)
                       
                       # Display list of high rated movies
                       st.subheader("Movies in this Category")
                       display_rated_movie_list("high")
                   else:
                       st.info("You haven't rated any movies in this category yet.")
               else:
                   st.info("You haven't rated any movies in this category yet.")

       # Add rating functionality directly to this page
       st.subheader("Rate More Movies")
       main_movie_query = st.text_input("Search for a movie to rate:", key="main_movie_search")
       
       if main_movie_query:
           search_results = search_movies(main_movie_query)
           if search_results:
               cols = st.columns(3)
               for i, movie in enumerate(search_results[:6]):  # Show up to 6 results
                   col_idx = i % 3
                   with cols[col_idx]:
                       st.write(f"**{movie['title']}**")
                       if movie.get('poster_path'):
                           st.image(f"https://image.tmdb.org/t/p/w154{movie['poster_path']}")
                       
                       # Store movie in rated_movies if not already present
                       if movie['id'] not in st.session_state.rated_movies:
                           st.session_state.rated_movies[movie['id']] = movie
                       
                       # Show current rating if it exists
                       current_rating = st.session_state.ratings.get(movie['id'], 3)
                       rating = st.slider(
                           "Rating", 
                           1, 5, current_rating, 
                           key=f"main_rating_{movie['id']}"
                       )
                       
                       if st.button("Submit Rating", key=f"main_submit_{movie['id']}"):
                           st.session_state.ratings[movie['id']] = rating
                           st.success(f"Rating for '{movie['title']}' submitted!")
                           st.experimental_rerun()
           else:
               st.info("No movies found with that title.")

       # Add a section for rating statistics
       st.subheader("Rating Statistics")
       col1, col2, col3 = st.columns(3)
       with col1:
           st.metric("Good Movies (4-5 ⭐)", high_count)
       with col2:
           st.metric("Average Movies (3 ⭐)", avg_count)
       with col3:
           st.metric("Bad Movies (1-2 ⭐)", low_count)
       
       # Genre preference analysis if there are enough ratings
       if high_count > 0:
           st.subheader("Genre Preferences Analysis")
           analyze_genre_preferences()


def create_combined_rating_graph():
    """Create a graph showing all rated movies connected to their rating categories"""
    G = nx.Graph()
    
    # Define rating categories
    category_labels = {
        "low": "Low Rated Movies (1-2 ⭐)",
        "average": "Average Rated Movies (3 ⭐)",
        "high": "High Rated Movies (4-5 ⭐)"
    }
    
    # Add central nodes for each category
    G.add_node(category_labels["low"], type='category', size=600, category="low")
    G.add_node(category_labels["average"], type='category', size=600, category="average")
    G.add_node(category_labels["high"], type='category', size=600, category="high")
    
    # Add movie nodes and connect to appropriate category
    for movie_id, rating in st.session_state.ratings.items():
        if movie_id in st.session_state.rated_movies:
            movie = st.session_state.rated_movies[movie_id]
            movie_title = movie['title']
            
            # Determine category
            if rating <= 2:
                category = "low"
            elif rating == 3:
                category = "average"
            else:  # rating >= 4
                category = "high"
            
            # Add movie node
            G.add_node(movie_title, type='movie', rating=rating, size=200, category=category)
            # Connect to category center
            G.add_edge(movie_title, category_labels[category], weight=1)
    
    # Connect movies with similar genres
    movie_genres = {}
    for movie_id in st.session_state.ratings:
        if movie_id in st.session_state.rated_movies:
            movie = st.session_state.rated_movies[movie_id]
            if 'genre_ids' in movie:
                movie_genres[movie['title']] = set(movie.get('genre_ids', []))
    
    # Connect movies by common genres (but with lower weight than category connections)
    for title1, genres1 in movie_genres.items():
        for title2, genres2 in movie_genres.items():
            if title1 != title2:
                common_genres = genres1.intersection(genres2)
                if common_genres:
                    G.add_edge(title1, title2, weight=min(len(common_genres)/3, 0.5))
    
    return G


def visualize_combined_graph(G):
    """Visualize a combined graph with all rating categories"""
    plt.figure(figsize=(12, 10))
    plt.title("All Movie Ratings")
    
    # Use ForceAtlas2-like layout for better separation
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Filter nodes by type and category
    category_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'category']
    
    # Movies by category
    low_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'movie' and G.nodes[node].get('category') == 'low']
    avg_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'movie' and G.nodes[node].get('category') == 'average']
    high_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'movie' and G.nodes[node].get('category') == 'high']
    
    # Draw category nodes
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[node for node in category_nodes if G.nodes[node].get('category') == 'low'],
                         node_color='#F44336',  # Red
                         node_size=[G.nodes[node].get('size', 600) for node in category_nodes if G.nodes[node].get('category') == 'low'],
                         alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[node for node in category_nodes if G.nodes[node].get('category') == 'average'],
                         node_color='#FFC107',  # Yellow
                         node_size=[G.nodes[node].get('size', 600) for node in category_nodes if G.nodes[node].get('category') == 'average'],
                         alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[node for node in category_nodes if G.nodes[node].get('category') == 'high'],
                         node_color='#4CAF50',  # Green
                         node_size=[G.nodes[node].get('size', 600) for node in category_nodes if G.nodes[node].get('category') == 'high'],
                         alpha=0.8)
    
    # Draw movie nodes by category
    if low_nodes:
        nx.draw_networkx_nodes(G, pos,
                              nodelist=low_nodes,
                              node_color='#FFCDD2',  # Light red
                              node_size=[G.nodes[node].get('size', 200) for node in low_nodes],
                              alpha=0.7)
    
    if avg_nodes:
        nx.draw_networkx_nodes(G, pos,
                              nodelist=avg_nodes,
                              node_color='#FFF9C4',  # Light yellow
                              node_size=[G.nodes[node].get('size', 200) for node in avg_nodes],
                              alpha=0.7)
    
    if high_nodes:
        nx.draw_networkx_nodes(G, pos,
                              nodelist=high_nodes,
                              node_color='#A5D6A7',  # Light green
                              node_size=[G.nodes[node].get('size', 200) for node in high_nodes],
                              alpha=0.7)
    
    # Draw primary edges (to category centers) with stronger lines
    category_edges = [(u, v) for u, v in G.edges() if 
                     (u in category_nodes or v in category_nodes)]
    nx.draw_networkx_edges(G, pos, 
                          edgelist=category_edges,
                          width=2.0, 
                          alpha=0.7, 
                          edge_color='#555555')
    
    # Draw genre similarity edges with thinner lines
    genre_edges = [(u, v) for u, v in G.edges() if 
                  u not in category_nodes and v not in category_nodes]
    if genre_edges:
        edge_weights = [G[u][v]['weight'] * 2 for u, v in genre_edges]
        nx.draw_networkx_edges(G, pos, 
                              edgelist=genre_edges,
                              width=edge_weights, 
                              alpha=0.3, 
                              edge_color='gray')
    
    # Draw labels with different sizes
    category_labels = {node: node for node in category_nodes}
    
    # Only show movie titles for nodes with space around them
    # This prevents overcrowding in the visualization
    degree_dict = dict(G.degree())
    movie_labels = {}
    for node in G.nodes():
        if G.nodes[node].get('type') == 'movie':
            # Only label movies with fewer connections for clarity
            if degree_dict[node] <= 3:  
                movie_labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=category_labels, font_size=12, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=movie_labels, font_size=8)
    
    plt.axis('off')
    return plt


def display_all_rated_movies():
    """Display all rated movies grouped by rating category"""
    # Create 3 columns - one for each rating category
    cols = st.columns(3)
    
    with cols[0]:
        st.subheader("Low Rated (1-2 ⭐)")
        display_rated_movie_list_compact("low")
    
    with cols[1]:
        st.subheader("Average Rated (3 ⭐)")
        display_rated_movie_list_compact("average")
    
    with cols[2]:
        st.subheader("High Rated (4-5 ⭐)")
        display_rated_movie_list_compact("high")


def display_rated_movie_list_compact(rating_category):
    """Display a compact list of movies in the specified rating category"""
    # Filter movies by rating category
    movies = []
    for movie_id, rating in st.session_state.ratings.items():
        if movie_id in st.session_state.rated_movies:
            movie = st.session_state.rated_movies[movie_id]
            if (rating_category == "low" and rating <= 2) or \
               (rating_category == "average" and rating == 3) or \
               (rating_category == "high" and rating >= 4):
                movies.append({**movie, 'user_rating': rating})
    
    # Sort by rating (descending) then title
    movies.sort(key=lambda x: (-x['user_rating'], x['title']))
    
    # Display as a compact list
    for movie in movies:
        st.write(f"**{movie['title']}** ({'⭐' * movie['user_rating']})")
        
        # Re-rate option in an expander
        with st.expander("Re-rate or view details"):
            if movie.get('poster_path'):
                st.image(f"https://image.tmdb.org/t/p/w154{movie['poster_path']}")
            
            # Re-rating slider
            new_rating = st.slider(
                "Rating", 
                1, 5, movie['user_rating'], 
                key=f"rerate_{movie['id']}"
            )
            
            # Update button
            if st.button("Update Rating", key=f"update_{movie['id']}"):
                st.session_state.ratings[movie['id']] = new_rating
                st.success(f"Rating updated for '{movie['title']}'!")
                st.experimental_rerun()
            
            # Display genres if available
            if 'genre_ids' in movie:
                genre_dict = fetch_genre_list()
                genres = [genre_dict.get(gid, "Unknown") for gid in movie['genre_ids'] if gid in genre_dict]
                if genres:
                    st.write(f"Genres: {', '.join(genres)}")
                    
# Run the app
if __name__ == "__main__":
    main()