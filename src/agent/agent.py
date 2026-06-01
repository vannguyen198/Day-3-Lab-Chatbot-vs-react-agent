import re
from typing import List, Dict, Any
from src.core.llm_provider import LLMProvider
from src.telemetry.metrics import tracker

class ReActAgent:
    """
    SKELETON: A ReAct-style Agent that follows the Thought-Action-Observation loop.
    Students should implement the core loop logic and tool execution.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self) -> str:
        """
        Generate the system prompt that guides the agent's behavior.
        This should include instructions on how to format Thought, Action, and Observation.
        """
        tool_list = "\n".join([f"{tool['name']}: {tool['description']}" for tool in self.tools])
        system_prompt = (
            "You are a ReAct agent. Follow this format:\n"
            "Thought: <your thought process>\n"
            "Action: <tool_name(arguments)>\n"
            "Observation: <result of action>\n"
            "Final Answer: <your final answer if you have one>\n\n"
            f"Available tools:\n{tool_list}\n\n"
        )
        return system_prompt

    def run(self, user_input: str) -> str:
        """
        Main loop for the ReAct agent.
        1.  Generate Thought and Action using the LLM.
        2.  Execute the Action (tool call) and get Observation.
        3.  Repeat until a Final Answer is produced or max steps reached.
        """
        self.history.append(f"User: {user_input}")
        system_prompt = self.get_system_prompt() 
        for _ in range(self.max_steps):
            llm_input = system_prompt + "\n".join(self.history)
            response = self.llm.generate(llm_input)

            response_text = response.get("content") if isinstance(response, dict) else str(response)
            response_text = response_text.strip()
            print("\n=== MODEL RESPONSE ===")
            print(response_text)
            print("======================\n")
            self.history.append(f"Agent: {response_text}")

            if isinstance(response, dict):
                tracker.track_request(
                    provider=response.get("provider", "unknown"),
                    model=getattr(self.llm, "model_name", "unknown"),
                    usage=response.get("usage", {}),
                    latency_ms=response.get("latency_ms", 0)
                )

            # Parse Action and Final Answer
            action_match = re.search(r'Action:\s*(.*)', response_text, re.IGNORECASE)
            final_answer_match = re.search(r'Final Answer:\s*(.*)', response_text, re.IGNORECASE)

            if final_answer_match:
                return final_answer_match.group(1).strip()

            if action_match:
                action_str = action_match.group(1).strip()
                tool_name, args = self.parse_action(action_str)
                observation = self._execute_tool(tool_name, args)
                self.history.append(f"Observation: {observation}")
                self.history.append("The observation contains the answer. Respond with Final Answer.")
        
        return "Max steps reached without a final answer."


    def parse_action(self, action_str: str) -> tuple:
        """
        Parse action string in format: tool_name(arguments) or tool_name.
        Supports optional quoted string arguments, empty arguments, and no-argument calls.
        Returns: (tool_name, args)
        """
        if not action_str:
            return None, None

        match = re.match(r'^\s*([A-Za-z_]\w*)\s*(?:\(\s*(.*)\s*\))?\s*$', action_str)
        if match:
            tool_name = match.group(1)
            args = match.group(2) or ""
            args = args.strip()

            if len(args) >= 2 and ((args.startswith('"') and args.endswith('"')) or (args.startswith("'") and args.endswith("'"))):
                args = args[1:-1]
            return tool_name, args
        return None, None
    
    @staticmethod
    def get_weather_function(city: str) -> str:
        weather_data = {
            "Hanoi": {"temp": "35°C", "condition": "sunny"},
            "London": {"temp": "20°C", "condition": "cloudy"},
            "Tokyo": {"temp": "28°C", "condition": "humid"},
            "New York": {"temp": "22°C", "condition": "windy"},
            "Paris": {"temp": "25°C", "condition": "clear"},
            "Sydney": {"temp": "18°C", "condition": "rainy"},
            "Mumbai": {"temp": "30°C", "condition": "hot"},
            "Los Angeles": {"temp": "27°C", "condition": "sunny"},
            "San Francisco": {"temp": "19°C", "condition": "foggy"},
            "Berlin": {"temp": "23°C", "condition": "partly cloudy"},
            "Moscow": {"temp": "15°C", "condition": "snowy"}
        }

        info = weather_data.get(city, {"temp": "25°C", "condition": "sunny"})
        return f"Weather in {city} is {info['condition']} with {info['temp']}."
    
    my_tools = [
        {
            "name": "weather", 
            "description": "Gets the weather for a city. Input: 'city_name'", 
            "func": get_weather_function
        }
    ]
    
    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Helper method to execute tools by name.
        """
        for tool in self.tools:
            if tool['name'] == tool_name:
                try:
                    # Execute the function associated with the tool
                    # 'args' is passed as a string from the LLM parser
                    result = tool['func'](args)
                    return str(result)
                except (ValueError, TypeError, KeyError) as e:
                    # Catch specific errors to prevent the agent loop from crashing
                    return f"Error executing {tool_name}: {str(e)}"
        return f"Tool {tool_name} not found."
