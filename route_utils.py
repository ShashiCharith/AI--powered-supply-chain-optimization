import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from folium import plugins
import networkx as nx
from typing import List, Dict, Tuple
import json

class RouteOptimizer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="supply_chain_optimizer")
        self.graph = nx.Graph()
        
    def get_coordinates(self, city: str) -> Tuple[float, float]:
        """Get coordinates for a city name."""
        try:
            location = self.geolocator.geocode(city + ", India")
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            print(f"Error getting coordinates for {city}: {str(e)}")
            return None
    
    def create_distance_matrix(self, locations: List[Tuple[float, float]]) -> np.ndarray:
        """Create a distance matrix between all locations."""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = geodesic(locations[i], locations[j]).kilometers
                    
        return matrix
    
    def optimize_route(self, 
                      locations: List[Tuple[float, float]], 
                      demands: List[float] = None,
                      priorities: List[int] = None,
                      weather_impact: float = 1.0,
                      traffic_weight: float = 1.0) -> List[int]:
        """Optimize route using a modified TSP algorithm with constraints."""
        n = len(locations)
        
        # Create distance matrix
        distance_matrix = self.create_distance_matrix(locations)
        
        # Apply weather and traffic impacts
        distance_matrix = distance_matrix * weather_impact * traffic_weight
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with demands and priorities
        for i in range(n):
            G.add_node(i, 
                      pos=locations[i],
                      demand=demands[i] if demands else 0,
                      priority=priorities[i] if priorities else 1)
        
        # Add edges with weights
        for i in range(n):
            for j in range(i+1, n):
                G.add_edge(i, j, weight=distance_matrix[i][j])
        
        # Use networkx's approximation algorithm for TSP
        route = nx.approximation.traveling_salesman_problem(G, cycle=False)
        
        return route
    
    def calculate_metrics(self, 
                         route: List[int], 
                         locations: List[Tuple[float, float]],
                         demands: List[float] = None,
                         fuel_price: float = 100.0) -> Dict:
        """Calculate route metrics."""
        total_distance = 0
        total_demand = 0
        
        for i in range(len(route)-1):
            start = locations[route[i]]
            end = locations[route[i+1]]
            distance = geodesic(start, end).kilometers
            total_distance += distance
            
            if demands:
                total_demand += demands[route[i]]
        
        # Estimate time (assuming average speed of 40 km/h)
        estimated_time = total_distance / 40  # hours
        
        # Calculate fuel cost (assuming 10 km/l fuel efficiency)
        fuel_cost = (total_distance / 10) * fuel_price
        
        return {
            "total_distance": round(total_distance, 2),
            "estimated_time": round(estimated_time, 2),
            "fuel_cost": round(fuel_cost, 2),
            "total_demand": round(total_demand, 2) if demands else None
        }
    
    def create_map(self, 
                  route: List[int], 
                  locations: List[Tuple[float, float]],
                  city_names: List[str]) -> folium.Map:
        """Create an interactive map with the optimized route."""
        # Create base map centered on India
        m = folium.Map(location=[20.5937, 78.9629], 
                      zoom_start=5,
                      tiles='CartoDB positron')
        
        # Add route lines
        route_coords = [locations[i] for i in route]
        folium.PolyLine(route_coords, 
                       weight=2, 
                       color='blue', 
                       opacity=0.8).add_to(m)
        
        # Add markers for each location
        for i, (lat, lon) in enumerate(locations):
            # Different colors for start, end, and intermediate points
            if i == route[0]:
                color = 'green'  # Start
            elif i == route[-1]:
                color = 'red'    # End
            else:
                color = 'blue'   # Intermediate
                
            folium.Marker(
                [lat, lon],
                popup=city_names[i],
                icon=folium.Icon(color=color)
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def process_delivery_data(self, df: pd.DataFrame) -> Tuple[List[Tuple[float, float]], List[str], List[float], List[int]]:
        """Process delivery data from DataFrame."""
        locations = []
        city_names = []
        demands = []
        priorities = []
        
        for _, row in df.iterrows():
            city = row['city']
            coords = self.get_coordinates(city)
            
            if coords:
                locations.append(coords)
                city_names.append(city)
                demands.append(row.get('demand', 0))
                priorities.append(row.get('priority', 1))
        
        return locations, city_names, demands, priorities 