from openai import OpenAI
import json
from datetime import datetime

client = OpenAI(base_url="https://a7657d0d699c.ngrok-free.app/v1", api_key="dummy")

# --- Tool Function Definitions ---

def get_weather(location: str, unit: str = "celsius"):
    """Gets the current weather for a given location."""
    print(f"--- Calling get_weather(location='{location}', unit='{unit}') ---")
    if "astana" in location.lower():
        return f"The weather in Astana is currently cool and clear at 8 degrees {unit}."
    else:
        return f"The weather in {location} is currently 20 degrees {unit}."

def schedule_meeting(participants: list, title: str, time: str, location: str = "Online", duration_minutes: int = 30):
    """
    Schedules a meeting with a list of participants at a specified time.
    Time should be in ISO 8601 format, e.g., 2025-09-16T14:00:00.
    """
    print(f"--- Calling schedule_meeting() ---")
    participant_str = ", ".join(participants)
    return (
        f"Meeting Scheduled Successfully!\n"
        f"\tTitle: {title}\n"
        f"\tTime: {time}\n"
        f"\tParticipants: {participant_str}\n"
        f"\tLocation: {location}\n"
        f"\tDuration: {duration_minutes} minutes"
    )

# --- Mapping tool names to the actual Python functions ---
tool_functions = {
    "get_weather": get_weather,
    "schedule_meeting": schedule_meeting
}

# --- Tool Schemas for the Model ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"] # Note: I made 'unit' optional here
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedules a calendar event or meeting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of people to invite to the meeting."
                    },
                    "title": {
                        "type": "string",
                        "description": "The title or subject of the meeting."
                    },
                    "time": {
                        "type": "string",
                        "description": "The start time of the meeting in ISO 8601 format, e.g., YYYY-MM-DDTHH:MM:SS"
                    },
                    "location": {
                        "type": "string",
                        "description": "The location of the meeting. Defaults to 'Online' if not specified."
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "The duration of the meeting in minutes. Defaults to 30."
                    }
                },
                "required": ["participants", "title", "time"]
            }
        }
    }
]

# --- User Prompt ---
# A natural language prompt that contains all the necessary info for the new tool.
# Since today is September 15th, "tomorrow at 2 PM" should be resolved to Sept 16th.
messages = [
    {"role": "user", "content": "Hey, can you schedule a meeting for me and Vladimir about the 'Qwen3 Project Plan' for tomorrow at 2 PM? It should be 45 minutes long and take place in the main conference room."}
]

print("Sending request to the model...")
response = client.chat.completions.create(
    model="kita", # Use your served model name
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# --- Robustly handle the response ---
message = response.choices[0].message

if message.tool_calls:
    print("✅ Model decided to call a tool.")
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    
    if function_name in tool_functions:
        arguments = json.loads(tool_call.function.arguments)
        print(f"Function to call: {function_name}")
        print(f"Arguments: {arguments}")
        
        function_to_call = tool_functions[function_name]
        result = function_to_call(**arguments)
        
        print("\n--- Result ---")
        print(result)
    else:
        print(f"⚠️ Model tried to call an unknown function: {function_name}")

else:
    print("❌ Model responded with text instead of a tool call.")
    print(f"   Model Response: {message.content}")