from datetime import datetime, timedelta
import ee
import streamlit as st
import geemap.foliumap as geemap
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re

load_dotenv()

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

client = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def get_land_use_analysis(query):
    prompt = f"""
    Analyze the land use patterns and changes for the location and time period specified in the following query:

    {query}

    Provide a detailed response that includes:
    1. A summary of significant changes in land use patterns over the specified time period.
    2. Statistical data from reputable sources, including percentages of land use changes if available.
    3. Key trends in urbanization, agricultural land conversion, and environmental impact.
    4. Any relevant government policies or initiatives that have influenced these changes.
    5. Citations or references to specific studies or reports that support your analysis.

    Format your response with clear headings, bullet points for key information, and a section for additional resources or references.

    Ensure that all information provided is factual and based on verifiable data or reputable studies. If specific data for the exact time period is not available, use the closest available data and note this in your response.
    """
    response = client.invoke(prompt)
    return response.content

def extract_attributes(query):
    prompt = f"""
    Extract the following attributes from the given query:
    1. Longitude
    2. Latitude
    3. Zoom level (default to 11 if not specified)
    4. Start date
    5. End date

    Query: {query}

    Respond ONLY with a valid JSON object using this exact format:
    {{
        "longitude": <float>,
        "latitude": <float>,
        "zoom": <int>,
        "start_date": "<YYYY-MM-DD>",
        "end_date": "<YYYY-MM-DD>"
    }}
    Use your best judgment to infer values if they're not explicitly stated. """
    response = client.invoke(prompt)
    
    try:
        attributes = json.loads(response.content) 
    except json.JSONDecodeError:
        print("JSON parsing failed. Falling back to regex extraction.")
        attributes = {
            "longitude": float(re.search(r'"longitude":\s*([-\d.]+)', response.content).group(1)),
            "latitude": float(re.search(r'"latitude":\s*([-\d.]+)', response.content).group(1)),
            "zoom": int(re.search(r'"zoom":\s*(\d+)', response.content).group(1)),
            "start_date": re.search(r'"start_date":\s*"(\d{4}-\d{2}-\d{2})"', response.content).group(1),
            "end_date": re.search(r'"end_date":\s*"(\d{4}-\d{2}-\d{2})"', response.content).group(1)
        }

    # Convert dates to datetime objects
    attributes['start_date'] = datetime.strptime(attributes['start_date'], '%Y-%m-%d').date()
    attributes['end_date'] = datetime.strptime(attributes['end_date'], '%Y-%m-%d').date()

    # Set default values if extraction fails
    attributes.setdefault('longitude', 78.4867)  # Default to Hyderabad coordinates
    attributes.setdefault('latitude', 17.3850)
    attributes.setdefault('zoom', 11)
    today = datetime.now().date()
    attributes.setdefault('start_date', today - timedelta(days=3650))  # Default to 10 years ago
    attributes.setdefault('end_date', today)

    return attributes

# Streamlit app
st.set_page_config(layout="wide")
st.title("Comparing Global Land Cover Maps")

# Get user query
user_query = st.text_input("Enter your query:")

if user_query:
    # Extract attributes from the query
    attributes = extract_attributes(user_query)

    analysis = get_land_use_analysis(user_query)
    st.markdown(analysis)

    # Use the extracted attributes in your Streamlit app
    col1, col2 = st.columns([4, 1])

    Map = geemap.Map()
    Map.add_basemap("ESA WorldCover 2020 S2 FCC")
    Map.add_basemap("ESA WorldCover 2020 S2 TCC")
    Map.add_basemap("HYBRID")

    esa = ee.ImageCollection("ESA/WorldCover/v100").first()
    esa_vis = {"bands": ["Map"]}

    esri = ee.ImageCollection(
        "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m"
    ).mosaic()
    esri_vis = {
        "min": 1,
        "max": 10,
        "palette": [
            "#1A5BAB", "#358221", "#A7D282", "#87D19E", "#FFDB5C",
            "#EECFA8", "#ED022A", "#EDE9E4", "#F2FAFF", "#C8C8C8",
        ],
    }

    with col2:
        longitude = st.number_input("Longitude", -180.0, 180.0, attributes['longitude'])
        latitude = st.number_input("Latitude", -90.0, 90.0, attributes['latitude'])
        zoom = st.number_input("Zoom", 0, 20, attributes['zoom'])

        Map.setCenter(longitude, latitude, zoom)

        start = st.date_input("Start Date for Dynamic World", attributes['start_date'])
        end = st.date_input("End Date for Dynamic World", attributes['end_date'])

        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

        region = ee.Geometry.BBox(-179, -89, 179, 89)
        try:
            dw = geemap.dynamic_world(region, start_date, end_date, return_type="visualize")
            dw_layer = geemap.ee_tile_layer(dw, {}, "Dynamic World Land Cover")
        except Exception as e:
            st.warning(f"Error loading Dynamic World layer: {str(e)}")
            dw_layer = None

        layers = {
            "Dynamic World": dw_layer,
            "ESA Land Cover": geemap.ee_tile_layer(esa, esa_vis, "ESA Land Cover"),
            "ESRI Land Cover": geemap.ee_tile_layer(esri, esri_vis, "ESRI Land Cover"),
        }

        # Remove None layers
        layers = {k: v for k, v in layers.items() if v is not None}

        options = list(layers.keys())
        left = st.selectbox("Select a left layer", options, index=1)
        right = st.selectbox("Select a right layer", options, index=0)

        left_layer = layers[left]
        right_layer = layers[right]

        Map.split_map(left_layer, right_layer)

        legend = st.selectbox("Select a legend", options, index=options.index(right))
        if legend == "Dynamic World":
            Map.add_legend(
                title="Dynamic World Land Cover",
                builtin_legend="Dynamic_World",
            )
        elif legend == "ESA Land Cover":
            Map.add_legend(title="ESA Land Cover", builtin_legend="ESA_WorldCover")
        elif legend == "ESRI Land Cover":
            Map.add_legend(title="ESRI Land Cover", builtin_legend="ESRI_LandCover")

    with col1:
        Map.to_streamlit(height=750)
    
    
