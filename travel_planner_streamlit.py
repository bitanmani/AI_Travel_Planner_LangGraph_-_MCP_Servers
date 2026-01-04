"""
AI Travel Planner - Streamlit Application
A multi-agent AI system for comprehensive travel planning using LangGraph and Gemini.
"""

import streamlit as st
import os
import json
import time
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional, Sequence, Annotated
import warnings
import operator

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

# Utility imports
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool
def web_search_tool(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    """Search the web using DuckDuckGo."""
    try:
        from ddgs import DDGS
    except Exception:
        from duckduckgo_search import DDGS
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", ""))
            })
        
        return formatted_results
    except Exception as e:
        return [{"title": "Search Error", "url": "", "snippet": f"Error: {str(e)}"}]

@tool
def get_weather_forecast(destination: str, start_date: str, num_days: int = 7) -> Dict[str, Any]:
    """Get weather forecast using Open-Meteo API."""
    try:
        # Geocode
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_response = requests.get(geocode_url, params={"name": destination, "count": 1}, timeout=10)
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return {"error": f"Location not found: {destination}"}
        
        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        
        # Weather forecast
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = (start + timedelta(days=num_days - 1)).strftime('%Y-%m-%d')
        
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode,precipitation_probability_max",
            "timezone": "auto",
            "start_date": start.strftime('%Y-%m-%d'),
            "end_date": end
        }
        
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_data = weather_response.json()
        
        daily = weather_data.get("daily", {})
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 95: "Thunderstorm"
        }
        
        forecast = []
        for i in range(len(daily.get("time", []))):
            forecast.append({
                "date": daily["time"][i],
                "temp_max_c": daily["temperature_2m_max"][i],
                "temp_min_c": daily["temperature_2m_min"][i],
                "conditions": weather_codes.get(daily["weathercode"][i], "Unknown"),
                "precipitation_prob": daily["precipitation_probability_max"][i]
            })
        
        return {
            "location": f"{location.get('name')} ({lat:.2f}°, {lon:.2f}°)",
            "forecast": forecast,
            "num_days": len(forecast)
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# STATE DEFINITION
# =============================================================================

class TravelPlannerState(TypedDict):
    destination: str
    num_days: int
    travel_style: str
    budget_range: str
    start_date: str
    interests: list
    headcount: int
    multi_city: bool
    cities: list
    research_results: str
    weather_analysis: str
    hotel_recommendations: str
    budget_estimate: str
    logistics_plan: str
    final_itinerary: str
    activity_bookings: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_step: str
    revision_count: int
    workflow_start_time: float
    workflow_end_time: float
    total_cost_estimate: float
    errors: list

# =============================================================================
# AGENT CLASS
# =============================================================================

class TravelAgent:
    def __init__(self, name: str, role: str, system_prompt: str, 
                 api_key: str, temperature: float = 0.7):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=8000,
            timeout=120
        )
    
    def invoke(self, messages: List, max_retries: int = 2) -> Dict[str, Any]:
        start_time = time.time()
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(full_messages)
                elapsed = time.time() - start_time
                return {
                    "agent": self.name,
                    "content": response.content,
                    "elapsed_time": elapsed,
                    "attempt": attempt + 1
                }
            except Exception as e:
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)

# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def create_agents(api_key: str) -> Dict:
    """Create all specialized agents."""
    
    research_prompt = """You are a travel research expert. Find top attractions, restaurants, 
    and accommodations. Keep responses concise and actionable."""
    
    weather_prompt = """You are a weather analyst. Provide weather forecasts, packing 
    recommendations, and activity suggestions based on weather."""
    
    hotel_prompt = """You are an accommodation specialist. Find hotels matching budget and 
    preferences. Provide booking links and neighborhood recommendations."""
    
    budget_prompt = """You are a budget expert. Calculate realistic trip costs with detailed 
    breakdowns. Include daily budgets and money-saving tips."""
    
    logistics_prompt = """You are a logistics expert. Plan efficient routes and transportation. 
    Suggest local transit options and intercity connections."""
    
    planner_prompt = """You are a master itinerary planner. Create detailed day-by-day schedules 
    with specific times, locations, and practical tips."""
    
    activities_prompt = """You are an activities specialist. Find booking links for tours and 
    attractions on major platforms like Viator and GetYourGuide."""
    
    return {
        "research": TravelAgent("ResearchAgent", "Research", research_prompt, api_key, 0.6),
        "weather": TravelAgent("WeatherAgent", "Weather", weather_prompt, api_key, 0.5),
        "hotel": TravelAgent("HotelAgent", "Hotels", hotel_prompt, api_key, 0.6),
        "budget": TravelAgent("BudgetAgent", "Budget", budget_prompt, api_key, 0.5),
        "logistics": TravelAgent("LogisticsAgent", "Logistics", logistics_prompt, api_key, 0.6),
        "planner": TravelAgent("PlannerAgent", "Planner", planner_prompt, api_key, 0.7),
        "activities": TravelAgent("ActivitiesAgent", "Activities", activities_prompt, api_key, 0.6)
    }

def research_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Research destination information."""
    if state.get("revision_count") is None:
        state["revision_count"] = 0
    
    prompt = f"""Research {state['destination']} for {state['num_days']} days.
    Style: {state['travel_style']}, Budget: {state['budget_range']}, 
    Travelers: {state['headcount']}, Interests: {', '.join(state['interests'])}
    Find top attractions, dining, accommodations, and local tips."""
    
    response = agents["research"].invoke([HumanMessage(content=prompt)])
    state["research_results"] = response["content"]
    state["current_step"] = "research_complete"
    return state

def weather_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Analyze weather and provide recommendations."""
    prompt = f"""Analyze weather for {state['destination']} from {state['start_date']} 
    for {state['num_days']} days. Provide daily summary, packing list, and activity suggestions."""
    
    response = agents["weather"].invoke([HumanMessage(content=prompt)])
    state["weather_analysis"] = response["content"]
    state["current_step"] = "weather_complete"
    return state

def hotel_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Find hotel recommendations."""
    checkout = (datetime.strptime(state['start_date'], '%Y-%m-%d') + 
                timedelta(days=state['num_days'])).strftime('%Y-%m-%d')
    
    retry_instruction = ""
    if state.get("revision_count", 0) > 0:
        retry_instruction = "BROADEN your search to find any available accommodations."
    
    prompt = f"""Find accommodations for {state['destination']}.
    Check-in: {state['start_date']}, Check-out: {checkout}
    Guests: {state['headcount']}, Budget: {state['budget_range']}
    {retry_instruction}
    Provide 3-5 hotel recommendations with booking links."""
    
    response = agents["hotel"].invoke([HumanMessage(content=prompt)])
    state["hotel_recommendations"] = response["content"]
    state["current_step"] = "hotel_complete"
    return state

def budget_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Calculate trip budget."""
    prompt = f"""Estimate budget for {state['destination']} - {state['num_days']} days, 
    {state['headcount']} people, {state['budget_range']} budget.
    Provide daily breakdown and total cost estimate."""
    
    response = agents["budget"].invoke([HumanMessage(content=prompt)])
    state["budget_estimate"] = response["content"]
    
    # Extract cost estimate
    import re
    cost_match = re.search(r'\$[\d,]+', response["content"])
    if cost_match:
        cost_str = cost_match.group().replace("$", "").replace(",", "")
        try:
            state["total_cost_estimate"] = float(cost_str)
        except:
            state["total_cost_estimate"] = 0.0
    
    state["current_step"] = "budget_complete"
    return state

def logistics_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Plan transportation and routes."""
    if state["multi_city"]:
        prompt = f"""Plan multi-city logistics: {' → '.join(state['cities'])}
        Duration: {state['num_days']} days. Find transportation between cities and local options."""
    else:
        prompt = f"""Plan local logistics for {state['destination']}.
        Suggest best transportation, transit passes, and routing tips."""
    
    response = agents["logistics"].invoke([HumanMessage(content=prompt)])
    state["logistics_plan"] = response["content"]
    state["current_step"] = "logistics_complete"
    return state

def planner_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Create final itinerary."""
    hotels_content = state.get("hotel_recommendations", "").lower()
    revision_count = state.get("revision_count", 0)
    
    if "unavailable" in hotels_content and revision_count < 1:
        state["final_itinerary"] = "REVISE_HOTEL"
        state["revision_count"] = revision_count + 1
        return state
    
    prompt = f"""Create detailed {state['num_days']}-day itinerary for {state['destination']}.
    
    Research: {state.get('research_results', '')[:1500]}
    Weather: {state.get('weather_analysis', '')[:800]}
    Hotels: {state.get('hotel_recommendations', '')[:800]}
    Budget: {state.get('budget_estimate', '')[:600]}
    Logistics: {state.get('logistics_plan', '')[:800]}
    
    Create day-by-day schedule with times, locations, costs, and practical tips."""
    
    response = agents["planner"].invoke([HumanMessage(content=prompt)])
    state["final_itinerary"] = response["content"]
    state["current_step"] = "planner_complete"
    return state

def activities_node(state: TravelPlannerState, agents: Dict) -> TravelPlannerState:
    """Find activity booking links."""
    if state.get("final_itinerary") == "REVISE_HOTEL":
        return state
    
    prompt = f"""Find booking links for activities in this itinerary:
    {state.get('final_itinerary', '')[:2000]}
    
    Find official websites and major platforms (Viator, GetYourGuide, etc.)."""
    
    response = agents["activities"].invoke([HumanMessage(content=prompt)])
    state["activity_bookings"] = response["content"]
    state["current_step"] = "activities_complete"
    return state

def finalize_node(state: TravelPlannerState) -> TravelPlannerState:
    """Finalize workflow."""
    state["workflow_end_time"] = time.time()
    state["current_step"] = "complete"
    return state

# =============================================================================
# WORKFLOW CREATION
# =============================================================================

def create_workflow(agents: Dict):
    """Create the LangGraph workflow."""
    
    def router_check(state: TravelPlannerState) -> str:
        if state.get("final_itinerary") == "REVISE_HOTEL":
            return "hotel"
        return "activities"
    
    workflow = StateGraph(TravelPlannerState)
    
    # Add nodes with agents passed as argument
    workflow.add_node("research", lambda s: research_node(s, agents))
    workflow.add_node("weather", lambda s: weather_node(s, agents))
    workflow.add_node("hotel", lambda s: hotel_node(s, agents))
    workflow.add_node("budget", lambda s: budget_node(s, agents))
    workflow.add_node("logistics", lambda s: logistics_node(s, agents))
    workflow.add_node("planner", lambda s: planner_node(s, agents))
    workflow.add_node("activities", lambda s: activities_node(s, agents))
    workflow.add_node("finalize", finalize_node)
    
    # Define edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "weather")
    workflow.add_edge("weather", "hotel")
    workflow.add_edge("hotel", "budget")
    workflow.add_edge("budget", "logistics")
    workflow.add_edge("logistics", "planner")
    
    workflow.add_conditional_edges(
        "planner",
        router_check,
        {"hotel": "hotel", "activities": "activities"}
    )
    
    workflow.add_edge("activities", "finalize")
    workflow.add_edge("finalize", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ AI Travel Planner</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Multi-Agent AI System with LangGraph & Gemini")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Google API key to continue")
            st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
            st.stop()
        
        st.success("✅ API Key configured")
        
        st.markdown("---")
        st.markdown("### 🤖 AI Agents")
        st.markdown("""
        - 🔍 **Research Agent**: Finds attractions & dining
        - 🌤️ **Weather Agent**: Analyzes forecasts
        - 🏨 **Hotel Agent**: Recommends accommodations
        - 💰 **Budget Agent**: Calculates costs
        - 🗺️ **Logistics Agent**: Plans routes
        - 📋 **Planner Agent**: Creates itinerary
        - 🎫 **Activities Agent**: Finds bookings
        """)
    
    # Main form
    st.header("📝 Trip Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        destination = st.text_input(
            "🌍 Destination",
            value="Paris, France",
            help="Enter your destination city or region"
        )
        
        num_days = st.slider(
            "📅 Number of Days",
            min_value=1,
            max_value=30,
            value=5,
            help="How many days will you travel?"
        )
        
        headcount = st.number_input(
            "👥 Number of Travelers",
            min_value=1,
            max_value=20,
            value=2
        )
        
        start_date = st.date_input(
            "🗓️ Start Date",
            value=datetime.today() + timedelta(days=30)
        )
    
    with col2:
        travel_style = st.selectbox(
            "🎨 Travel Style",
            ["Culture", "Adventure", "Relaxation", "Family-Friendly", "Luxury", "Budget Backpacking"]
        )
        
        budget_range = st.selectbox(
            "💵 Budget Range",
            ["Budget-Friendly", "Mid-Range", "Luxury"]
        )
        
        interests = st.multiselect(
            "❤️ Interests",
            ["History & Culture", "Food & Dining", "Nature & Outdoors", 
             "Art & Museums", "Shopping", "Nightlife", "Adventure Sports"],
            default=["History & Culture", "Food & Dining"]
        )
    
    # Multi-city option
    multi_city = st.checkbox("🏙️ Multi-City Trip")
    cities = []
    
    if multi_city:
        cities_input = st.text_input(
            "Enter cities (comma-separated)",
            value="Rome, Florence, Venice",
            help="List cities in order of visit"
        )
        cities = [city.strip() for city in cities_input.split(",")]
    
    # Generate button
    if st.button("🚀 Generate Travel Plan", type="primary"):
        if not api_key:
            st.error("❌ Please provide your Google API key in the sidebar")
            return
        
        # Initialize
        with st.spinner("🤖 Initializing AI agents..."):
            try:
                agents = create_agents(api_key)
                workflow = create_workflow(agents)
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
                return
        
        # Prepare state
        initial_state = {
            "destination": destination,
            "num_days": num_days,
            "travel_style": travel_style,
            "budget_range": budget_range,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "interests": interests,
            "headcount": headcount,
            "multi_city": multi_city,
            "cities": cities if multi_city else [destination],
            "research_results": "",
            "weather_analysis": "",
            "hotel_recommendations": "",
            "budget_estimate": "",
            "logistics_plan": "",
            "final_itinerary": "",
            "activity_bookings": "",
            "messages": [HumanMessage(content=f"Plan trip to {destination}")],
            "current_step": "initialized",
            "revision_count": 0,
            "workflow_start_time": time.time(),
            "workflow_end_time": 0.0,
            "total_cost_estimate": 0.0,
            "errors": []
        }
        
        config = {"configurable": {"thread_id": f"trip_{int(time.time())}"}}
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = ["research", "weather", "hotel", "budget", "logistics", "planner", "activities", "finalize"]
        current_step_idx = 0
        
        try:
            for output in workflow.stream(initial_state, config):
                for node_name, node_output in output.items():
                    if node_name != "__end__" and node_name in steps:
                        current_step_idx = steps.index(node_name) + 1
                        progress = current_step_idx / len(steps)
                        progress_bar.progress(progress)
                        status_text.text(f"✅ Completed: {node_name.title()}")
                        final_state = node_output
            
            # Display results
            progress_bar.progress(1.0)
            status_text.text("✅ All agents completed!")
            
            st.balloons()
            
            # Metrics
            st.header("📊 Trip Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Estimated Cost</h3>
                    <h2>${final_state.get('total_cost_estimate', 0):,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                elapsed = final_state['workflow_end_time'] - final_state['workflow_start_time']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⏱️ Processing Time</h3>
                    <h2>{elapsed:.1f}s</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔄 Revisions</h3>
                    <h2>{final_state.get('revision_count', 0)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Itinerary
            st.header("📋 Your Detailed Itinerary")
            with st.expander("View Full Itinerary", expanded=True):
                st.markdown(final_state.get('final_itinerary', 'No itinerary generated'))
            
            # Weather
            st.header("🌤️ Weather Forecast & Packing")
            with st.expander("View Weather Analysis"):
                st.markdown(final_state.get('weather_analysis', 'No weather data'))
            
            # Hotels
            st.header("🏨 Accommodation Recommendations")
            with st.expander("View Hotel Options"):
                st.markdown(final_state.get('hotel_recommendations', 'No hotels found'))
            
            # Activities
            st.header("🎫 Activity Bookings")
            with st.expander("View Booking Links"):
                st.markdown(final_state.get('activity_bookings', 'No activities found'))
            
            # Budget
            st.header("💵 Budget Breakdown")
            with st.expander("View Detailed Budget"):
                st.markdown(final_state.get('budget_estimate', 'No budget calculated'))
            
            # Logistics
            st.header("🚗 Transportation & Logistics")
            with st.expander("View Logistics Plan"):
                st.markdown(final_state.get('logistics_plan', 'No logistics plan'))
            
            # Download option
            st.header("💾 Export Your Plan")
            
            full_report = f"""
# {destination} Travel Plan
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Trip Details
- Destination: {destination}
- Duration: {num_days} days
- Travelers: {headcount}
- Budget: {budget_range}
- Style: {travel_style}
- Cost Estimate: ${final_state.get('total_cost_estimate', 0):,.2f}

## Itinerary
{final_state.get('final_itinerary', '')}

## Weather & Packing
{final_state.get('weather_analysis', '')}

## Accommodations
{final_state.get('hotel_recommendations', '')}

## Activities & Bookings
{final_state.get('activity_bookings', '')}

## Budget Breakdown
{final_state.get('budget_estimate', '')}

## Transportation
{final_state.get('logistics_plan', '')}
"""
            
            st.download_button(
                label="📥 Download Complete Plan (Markdown)",
                data=full_report,
                file_name=f"{destination.replace(' ', '_')}_travel_plan.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"❌ Error during planning: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
