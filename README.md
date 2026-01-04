# ✈️ AI Travel Planner: Multi-Agent System with LangGraph
🚀 Live Demo :
https://youtu.be/D4usPFlQ2A4

🌟 Overview
The AI Travel Planner is a cutting-edge application that leverages the power of multi-agent AI systems and Google's Gemini 2.0 to transform travel planning from a tedious, hours-long process into an automated, intelligent experience. Built on LangGraph, the system coordinates seven specialized AI agents that work together to research destinations, analyze weather, find accommodations, calculate budgets, plan logistics, and create detailed day-by-day itineraries.
Why This Matters
Traditional travel planning involves:

📱 Switching between dozens of websites
⏰ Hours of research and comparison
🧩 Manually piecing together disconnected information
💸 Uncertainty about total costs
🗓️ Trial and error with schedules and routes

Solution: A single interface where AI agents collaborate to deliver a complete travel plan in 2-5 minutes.

🎯 Key Features
🤖 Multi-Agent Collaboration
Seven specialized AI agents working in harmony:
| Agent              | Role                       | Key Capabilities                                                                 
|--------------------|----------------------------|----------------------------------------------------------------------------------
| 🔍 Research Agent   | Destination Research       | Web search, attraction discovery, restaurant recommendations                     
| 🌦️ Weather Agent    | Climate Analysis           | Real-time forecasts, packing lists, weather-appropriate activities               
| 🏨 Hotel Agent      | Accommodation Specialist   | Hotel search, booking platforms, neighborhood analysis                            
| 💰 Budget Agent     | Financial Planning         | Cost estimation, daily budgets, money-saving tips                                 
| 🚆 Logistics Agent  | Transportation Planning    | Route optimization, transit passes, multi-city connections                        
| 📅 Planner Agent    | Master Itinerary Creator   | Day-by-day schedules, time management, activity coordination                      
| 🎟️ Activities Agent | Booking Specialist         | Tour discovery, ticket links, platform recommendations                            

🔄 Intelligent Cyclic Workflow

Self-Correcting: If data quality is insufficient, agents can loop back and retry with improved strategies
Adaptive: The Planner Agent evaluates outputs and requests revisions when necessary
Resilient: Built-in retry logic and error handling ensure robust operation

🌍 Comprehensive Planning

✅ Single-city and multi-city itineraries
✅ Personalized based on interests (culture, food, adventure, art, etc.)
✅ Budget-conscious with three tiers: Budget-Friendly, Mid-Range, Luxury
✅ Weather-aware recommendations with daily forecasts
✅ Real-time web search for up-to-date information
✅ Direct booking links to Viator, GetYourGuide, TripAdvisor, etc.

📊 Rich Outputs

📅 Hour-by-hour itineraries with specific locations, costs, and durations
🌡️ Weather forecasts with packing recommendations
🏨 Hotel options across multiple booking platforms
💵 Detailed budget breakdowns per day and per person
🚗 Transportation guides including metro lines, walking routes, and intercity options
🎫 Curated activity bookings with direct purchase links

📖 Usage Guide
Basic Usage

Enter your Google API Key in the sidebar
Fill in trip details:

🌍 Destination (e.g., "Paris, France")
📅 Number of days (1-30)
👥 Number of travelers
🗓️ Start date
🎨 Travel style (Culture, Adventure, Luxury, etc.)
💵 Budget range
❤️ Interests (select multiple)


Enable multi-city (optional) and list cities
Click "Generate Travel Plan"
Wait 2-5 minutes while agents collaborate
Review your comprehensive travel plan
Download as Markdown for offline access

Advanced: Multi-City Planning
# Example: 10-day Italy tour
destination = "Italy (Rome & Florence & Venice)"
cities = ["Rome", "Florence", "Venice"]
num_days = 10
multi_city = True

# The system will:
# 1. Allocate days per city (e.g., 4-3-3)
# 2. Find intercity train/flight options
# 3. Create separate itineraries per city
# 4. Include travel days with lighter activities

Customization Options
Budget Ranges:

Budget-Friendly: $50-100/day per person
Mid-Range: $100-250/day per person
Luxury: $250-500+/day per person

Travel Styles:

Culture: Museums, historical sites, local traditions
Adventure: Hiking, water sports, extreme activities
Relaxation: Spas, beaches, slow-paced exploration
Family-Friendly: Kid-appropriate activities, safe areas
Luxury: Premium experiences, fine dining, boutique hotels
Budget Backpacking: Hostels, street food, free attractions

🔬 Technical Deep Dive
LangGraph State Machine
The system uses LangGraph's StateGraph to manage workflow:
class TravelPlannerState(TypedDict):
    # Input fields
    destination: str
    num_days: int
    interests: list
    
    # Agent outputs
    research_results: str
    weather_analysis: str
    final_itinerary: str
    
    # Control flow
    revision_count: int  # For cyclic behavior
    current_step: str

Key Innovation: Conditional Edges
workflow.add_conditional_edges(
    "planner",
    router_check,  # Evaluates data quality
    {
        "hotel": "hotel",          # Loop back if needed
        "activities": "activities"  # Proceed if satisfied
    }
)
Agent Design Pattern
Each agent follows a consistent structure:
class TravelAgent:
    def __init__(self, name, role, system_prompt, api_key, temperature):
        self.llm = ChatGoogleGenerativeAI(...)
        
    def invoke(self, messages, max_retries=2):
        # 1. Prepend system message
        # 2. Call Gemini with retry logic
        # 3. Return structured response
Temperature Settings:

Research/Weather/Logistics: 0.5-0.6 (factual accuracy)
Planner: 0.7 (balanced creativity)
Hotels/Activities: 0.6 (practical recommendations)

Tool Integration
Web Search (DuckDuckGo):
@tool
def web_search_tool(query: str, max_results: int = 6):
    # Returns: [{title, url, snippet}, ...]
    # Used by: Research, Hotel, Activities agents

Weather API (Open-Meteo):
@tool
def get_weather_forecast(destination: str, start_date: str, num_days: int):
    # Returns: {location, forecast: [{date, temp, conditions}]}
    # Used by: Weather agent

🎓 How It Works: Step-by-Step
Phase 1: Initialization (0.5s)

User submits form
Streamlit validates inputs
Agents are instantiated with API key
LangGraph workflow is compiled
Initial state is prepared

Phase 2: Sequential Execution (80-150s total)
Research Agent (5-10s)

Searches web for "{destination} travel tips"
Scrapes top travel blogs and official tourism sites
Extracts attractions, restaurants, accommodations
Outputs: 2-3 page research summary

Weather Agent (1-3s)

Calls Open-Meteo geocoding API
Fetches daily forecasts for trip dates
Analyzes temperature trends, precipitation
Outputs: Daily weather + packing list

Hotel Agent (5-15s)

Searches "{destination} {budget_range} hotels"
Identifies neighborhoods and booking platforms
Extracts price ranges and ratings
Outputs: 3-5 hotel recommendations + links
Note: If results are poor, Planner can loop back here

Budget Agent (3-8s)

Analyzes hotel prices from previous step
Estimates meal costs based on budget tier
Calculates transportation and activity costs
Adds 10-15% contingency buffer
Outputs: Daily breakdown + total estimate

Logistics Agent (8-15s)

For multi-city: searches train/flight options between cities
For single-city: researches metro/bus systems
Identifies transit passes and costs
Plans efficient routes between attractions
Outputs: Transportation guide with specific routes

Planner Agent (15-30s)

Evaluation Phase: Checks if hotel data is sufficient
Decision: Loop back to hotels OR proceed
Synthesis: Combines all agent outputs
Creates hour-by-hour schedules for each day
Balances activities with meals and rest
Considers weather and logistics constraints
Outputs: Complete day-by-day itinerary

Activities Agent (5-10s)

Parses itinerary for key activities
Searches booking platforms (Viator, GetYourGuide)
Finds official websites for major attractions
Outputs: Booking links with price estimates

Phase 3: Finalization (<1s)

Calculates total processing time
Compiles all outputs into structured format
Renders results in Streamlit UI
Enables download as Markdown

🙏 Acknowledgments

LangChain Team for the incredible LangGraph framework

Google for Gemini 2.0 and generous API access

Streamlit for making beautiful UIs effortless

Open-Meteo for free weather data

DuckDuckGo for privacy-respecting search
