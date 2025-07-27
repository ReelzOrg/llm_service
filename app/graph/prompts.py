from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

triage_prompt = ChatPromptTemplate.from_messages([
  ("system", """You are an expert at routing a user's request to the correct model provider and model.
  Based on the user's prompt, decide which provider and model to use.
  
  - For general conversation or simple tasks that can run locally, use 'ollama' with the 'gemma3:4b' model.
  - For coding specific tasks, use 'ollama' with the 'qwen2.5-coder:7b' model.
  """),
  ("human", "User prompt: {prompt}")
])

agent_prompt_template = ChatPromptTemplate.from_messages([
  ("system",
    "You are a helpful AI assistant. You have access to the following tools:\n\n"
    "{tools}\n\n" # The .bind_tools() method will populate this.
    "You must use the tools when you need current information. "
    "However, you must follow these critical rules:\n"
    "1. After calling a tool, you MUST analyze the result. \n"
    "2. If the tool's output is empty or does not contain the information you need, "
    "DO NOT try to call the same tool again. \n"
    "3. Instead, you MUST respond directly to the user, stating that you could not find the information "
    "and ask them to rephrase their question or provide more details.\n"
    "4. After a maximum of TWO tool attempts, you must give up and provide a final answer to the user, "
    "summarizing what you were able to find."
  ),
  MessagesPlaceholder(variable_name="messages")
])