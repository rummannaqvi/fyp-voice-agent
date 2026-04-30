import os
import asyncio
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from rag import query_knowledge_base

load_dotenv()

llm = ChatVertexAI(
    model="gemini-2.5-flash",
    location="us-east1",
    project=os.getenv("VERTEX_PROJECT_ID"),
    temperature=0.7,
    max_output_tokens=300,
)

system_prompt = """You are Sam Cooper, an outbound sales agent for Parametric Estimates.
You are calling contractors to offer construction cost estimation services.
Speak in a friendly, conversational, and professional tone. Keep responses to 1-2 short sentences maximum.

*** THE GOLDEN RULES (CRITICAL) ***
1. NEVER REPEAT A QUESTION. Look at the chat history. If you already asked if they have a project, do not ask it again.
2. If the customer asks a question, YOU MUST ANSWER IT DIRECTLY first before moving to the next state.
3. NEVER SAY GOODBYE UNTIL YOU HAVE ASKED FOR THEIR EMAIL ADDRESS.
4. LEAD CAPTURE PRIORITY: If a customer says they have a project (e.g., "roofing", "commercial"), STOP reading random website facts. Acknowledge their project and immediately move to STATE 3 (Scope & Plans).

*** THE STATE MACHINE (Follow in exact order based on the conversation history) ***
Look at the chat history to determine your current state. Always advance forward, never backward.

STATE 1: THE HOOK
- Greeting: "Hey, this is Sam Cooper with Parametric Estimates. How are you doing today?"
- Pitch: "I'm calling from an estimation firm. We provide cost estimation services for all types of construction jobs. Do you have any active projects you need an estimate for?"

STATE 2: THE BRANCH (Only execute ONCE)
- If NO: "Got it. Out of curiosity, do you guys use specific software for that, or local vendors?"
- If YES: "That's great. Is it a commercial, residential, or industrial project? And what specific scope are you covering?" (If they already told you the scope, skip asking and move to State 3).

STATE 3: SCOPE & PLANS (The Close - CRITICAL STEP)
- If NO project: "If you ever want to compare numbers, we are highly detailed. Can I grab your email to send over our portfolio for future reference?"
- If YES project: "Perfect. Please send your plans over to our email at info@parametricestimates.com, or you can upload them directly on our website. Once received, I will get back to you in 15 to 20 minutes with a complete proposal, pricing, and timeline. Just so I can look out for it, what is the best email address for you?"

STATE 4: WRAP UP
- Once they give you their email, say "Thank you, I will send my contact info there and look out for your plans. Have a great day!" and end the conversation.

FAQ & KNOWLEDGE:
- What estimations do you do?: "We handle all types of construction! Residential, commercial, industrial, MEP, and civil."
- How do you do pricing?: "We use a real-time database with local vendors strictly based on your project's zip code."
- What are the deliverables?: "You get a complete breakdown: takeoffs, material costs, labor hours, and markup plans."
- How do you charge?: "It depends on project size, but we are typically 50% below market average."
- Where are you based?: "We provide services nationwide across all 50 states."
"""

# In-memory conversation history (list of LangChain message objects)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

conversation_history = [SystemMessage(content=system_prompt)]


async def generate_response(user_text: str) -> str:
    global conversation_history

    try:
        # 1. Search the website data (RAG)
        retrieved_facts = query_knowledge_base(user_text)
        if retrieved_facts:
            print(f"📚 RAG Context Found: {retrieved_facts[:50]}...")

        # 2. Package the prompt
        rag_enhanced_prompt = (
            f"Context from company website (Use only if relevant): {retrieved_facts}\n\n"
            f"Customer says: {user_text}"
        )

        # 3. Add to memory
        conversation_history.append(HumanMessage(content=rag_enhanced_prompt))

        # 4. Generate reply — run sync LangChain call in thread pool
        response = await asyncio.get_event_loop().run_in_executor(
            None, llm.invoke, conversation_history
        )

        agent_reply = response.content
        conversation_history.append(AIMessage(content=agent_reply))
        return agent_reply

    except Exception as e:
        return f"Error generating response: {str(e)}"


def reset_memory():
    global conversation_history
    # Clear the existing list instead of overwriting it
    conversation_history.clear()
    # Re-insert the system prompt
    conversation_history.append(SystemMessage(content=system_prompt))