# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Le Van Nguyen
- **Student ID**: 2A202600569
- **Date**: June 1st, 2026

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implementated**: 
    - src/agent/agent.py:
        - Implemented and completed the ReAct loop logic.
        - Added action parsing and tool execution integration.
        - Added support for Observation handling and Final Answer generation.
    - tests/run_agent.py
        - Created and tested a ReAct agent runner using the local Phi-3 model.
        - Weather Tool Integration
    - Expanded the weather tool from a fixed response into a multi-city weather dataset with fallback handling (weather conditions).
- **Code Highlights**: 
1. ReAct Loop Implementation
```
for _ in range(self.max_steps):
    llm_input = system_prompt + "\n".join(self.history)
    response = self.llm.generate(llm_input)
    response_text = response.get("content") if isinstance(response, dict) else str(response)

    action_match = re.search(r'Action:\s*(.*)', response_text, re.IGNORECASE)
    final_answer_match = re.search(r'Final Answer:\s*(.*)', response_text, re.IGNORECASE)

    if final_answer_match:
        return final_answer_match.group(1).strip()

    if action_match:
        tool_name, args = self.parse_action(action_match.group(1))
        observation = self._execute_tool(tool_name, args)
        self.history.append(f"Observation: {observation}")
```
2. Action Parser
```
def parse_action(self, action_str: str) -> tuple:
    match = re.match(
        r'^\s*([A-Za-z_]\w*)\s*(?:\(\s*(.*)\s*\))?\s*$',
        action_str
    )

    if match:
        tool_name = match.group(1)
        args = match.group(2) or ""

        if len(args) >= 2 and (
            (args.startswith('"') and args.endswith('"')) or
            (args.startswith("'") and args.endswith("'"))
        ):
            args = args[1:-1]

        return tool_name, args

    return None, None
```
3. Weather Tool Enhancement
```
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

    info = weather_data.get(
        city,
        {"temp": "25°C", "condition": "sunny"}
    )

    return f"Weather in {city} is {info['condition']} with {info['temp']}."
```
4. Input as Prompt with PROVIDER_TYPE selection (run_agent.py)
```
PROVIDER_TYPE = "gemini" # Change to "local", "openai", or "gemini" as needed, local is default for testing

if PROVIDER_TYPE == "local":
    provider = LocalProvider(
        model_path="./models/Phi-3-mini-4k-instruct-q4.gguf"
    )

elif PROVIDER_TYPE == "openai":
    provider = OpenAIProvider(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY") #API key should be set in .env file
    )

elif PROVIDER_TYPE == "gemini":
    provider = GeminiProvider(
        model_name="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY") #API key should be set in .env file
    )

else:
    raise ValueError(f"Unknown provider: {PROVIDER_TYPE}")

agent = ReActAgent(
    llm=provider,
    tools=ReActAgent.my_tools
)

response = agent.run(
    "What's the weather in London?"
)

print(response)
```
- **Documentation**: 

The implemented code forms the core of the ReAct (Reasoning and Acting) workflow:
- The user submits a query.
- The LLM generates a Thought and Action.
- The Action is parsed by parse_action().
- The corresponding tool is executed through _execute_tool().
- The tool result is recorded as an Observation.
- The Observation is appended to the conversation history.
- The LLM uses the Observation to generate a Final Answer.
- The loop continues until a Final Answer is produced or the maximum number of steps is reached.

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**: 
The ReAct agent repeatedly called the weather tool and never generated a final response. The execution terminated after reaching the maximum number of reasoning steps (5).
- **Log Source**: `logs/2026-06-01.log`

Relevant telemetry entries (modified from log source by different lines):

LLM_METRIC
prompt_tokens: 95
completion_tokens: 40

LLM_METRIC
prompt_tokens: 153
completion_tokens: 40

LLM_METRIC
prompt_tokens: 212
completion_tokens: 41

LLM_METRIC
prompt_tokens: 271
completion_tokens: 40

LLM_METRIC
prompt_tokens: 329
completion_tokens: 41

The increasing prompt token count indicated that observations were being appended to the conversation history, but the model continued generating the same action (Generating output answers for weather).
- **Diagnosis**: 
The issue was not caused by tool execution because the weather tool successfully returned observations such as:

Observation: Weather in Hanoi is sunny with 35°C.

The issue was caused by the language model repeatedly choosing the weather tool even after receiving sufficient information.

The Phi-3 model correctly generated: Thought, Action, but failed to transition to the Final Answer after receiving the observation.

This was primarily a prompt-following and reasoning-loop issue rather than a tool or parser failure.
- **Solution**: 

Additional debugging output was added to inspect model responses and tool observations.

Example debugging output is as follows:
```
print("=== MODEL RESPONSE ===")
print(response_text)

print("Observation:", observation)
```

Example behavior:

Thought: To provide the user with the current weather in Hanoi, I need to use the weather tool.

Action: weather('Hanoi')

This same action was repeatedly generated, causing the output:

Max steps reached without a final answer.
---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1.  **Reasoning**: How did the `Thought` block help the agent compared to a direct Chatbot answer?

The Thought block allowed the agent to explicitly reason about the problem before taking action. Instead of directly generating an answer, the agent first determined whether a tool was needed and selected the appropriate action.

For example:
```
Thought: To provide the user with the current weather in Hanoi, I need to use the weather tool.

Action: weather('Hanoi')
```

2.  **Reliability**: In which cases did the Agent actually perform *worse* than the Chatbot?

The agent performed worse when the reasoning loop became stuck.

During testing, the Phi-3 model repeatedly generated the Action:
```
Action: weather('Hanoi')
```
even after receiving a valid observation. As a result, the agent reached the maximum number of steps without producing a final answer.

A chatbot would have responded immediately and avoided the loop entirely.

The agent also required:
- action parsing,
- tool execution,
- observation handling,
- prompt formatting,

which introduced additional failure points. Therefore, although agents are more powerful, they can be less reliable if the reasoning process is not carefully designed.
3.  **Observation**: How did the environment feedback (observations) influence the next steps?
Observations acted as feedback from the environment and allowed the agent to continue reasoning using real information obtained from tools.

For instance: At first:
```
Observation: Weather in Hanoi is sunny with 35°C.
```

After receiving this observation, the model had sufficient information to generate:
```
Final Answer: The weather in Hanoi is sunny with a temperature of 35°C.
```
Without observations, the model would have no way to incorporate tool results into its reasoning process, and it would keep generating logs only.

Observations therefore serve as the connection between external tools and the language model, enabling the agent to update its decisions based on newly acquired information.

---

## IV. Future Improvements (5 Points)

- **Scalability**: 
The current implementation executes tool calls sequentially inside the ReAct loop. For a production-level AI agent system, scalability could be improved by introducing asynchronous processing and task queues.

- Examples include:
    - Using asynchronous (async/await) tool execution.
    - Processing long-running tool calls through a message queue.
    - Deploying agent instances as independent services that can scale horizontally.
    - Supporting multiple concurrent user requests through containerized deployments

The example diagram is as follows:
```
User Request
↓
Agent Service
↓
Task Queue
↓
Tool Workers
↓
Observation
↓
Final Answer
```
This would allow the system to serve many users simultaneously without blocking on individual tool calls.

- **Safety**: 
The current ReAct agent executes tool calls directly once they are generated by the model. In a production environment, additional safety mechanisms should be introduced.

- Possible improvements:
    - Validate tool names and arguments before execution.
    - Restrict access to sensitive tools.
    - Add permission checks and rate limits.
    - Implement a Supervisor Agent that reviews actions before execution.
    - Log all actions and observations for auditing purposes.

- Sample flow:
```
Thought
↓
Proposed Action
↓
Supervisor Check
↓
Approved Tool Call
↓
Observation
```
This reduces the risk of incorrect or unsafe actions.

- **Performance**: 

The current system works well with a small number of tools, but production systems may contain hundreds of tools and knowledge sources.

- Possible improvements:
    - Validate tool names and arguments before execution.
    - Restrict access to sensitive tools.
    - Add permission checks and rate limits.
    - Implement a Supervisor Agent that reviews actions before execution.
    - Log all actions and observations for auditing purposes.
 
- Sample Flow:
```
User Query
↓
Vector Search
↓
Relevant Tools Selected
↓
ReAct Agent
↓
Answer
```
This would allow the system to serve many users simultaneously without blocking on individual tool calls.

Performance could be improved through the following:

- Vector databases for semantic tool retrieval.
- Caching frequently requested results.
- Model routing to select smaller or larger models depending on task complexity.
- Parallel tool execution where possible.
- Optimized telemetry and monitoring dashboards.

- Sample Flow:
```
User Query
↓
Vector Search
↓
Relevant Tools Selected
↓
ReAct Agent
↓
Answer
```
Rather than exposing all tools to the model, only the most relevant tools would be provided, reducing token usage and improving response speed.

## CONCLUSION:
The current implementation successfully demonstrates the core ReAct architecture using a local Phi-3 model and a weather tool. To scale the system for production use, asynchronous tool execution, safety validation layers, and intelligent tool retrieval mechanisms would be introduced. These improvements would increase reliability, efficiency, and security while maintaining the reasoning capabilities demonstrated in the laboratory environment.

---

