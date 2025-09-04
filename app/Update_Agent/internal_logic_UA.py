from ..classes import CompanyInfoMessage, UserInfoMessage 
from ..promptconfig import framework_themes_user, framework_themes_company, general_topic_info_summary, general_framework_info_user, general_framework_info_company
from pydantic_graph import GraphRunContext
import json
import logging
import re
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart
)
logging.basicConfig(level=logging.DEBUG)


####### TODO: Make it clearer what represents a good and bad score for some AI readiness dimension

# Define extraction prompt for the LLM.
extraction_system_prompt = (f"""
        I want you to act as an expert in extracting topic insights from an interview to create a profile of information. This profile can be seen as a pool of long-term memories.

        Your job is to analyze the conversation pair(s) provided and identify only the text segments from the user's messages that contain useful information about the topic, whether it’s about the user themselves or the company.

        Remember: “useful information” can be explicit statements (e.g., “I am responsible for distributing info about the topic”) or implicit indicators (e.g., “I don’t understand how this thing related to the topic works,” indicating a knowledge gap). 
        If the user expresses confusion, or requests for simplified explanations, that also signals their (lack of) skill or level in regards to the topic, and should be captured as a [User topic Info].

        If the segment pertains to the company's capabilities in regards to the topic, label it [Company topic Info].
        If the segment pertains to the user’s own capabilities in regards to the topic, label it [User topic Info].

        Output Format
        If multiple relevant segments exist, output each segment on a new line. Each line must be prefixed by the appropriate label in square brackets (i.e., [Company topic Info] or [User topic Info]). If there are no relevant segments, output nothing.

        Example
        User statement:
        “At our company we are really good at keeping up with new trends in the topic, and I’m the one who gathers all that info and shares it with my team.”

        Desired output:
        [Company topic Info] Good at keeping up with new trends in the topic
        [User topic Info] Gathers new info about the topic and shares it with his team

        Tips / Additional Guidance
        Look for statements about the user's topic proficiency, knowledge, or processes—for example, "I don’t understand what you mean with "X".", "we have no infrastructure, i guess", or "we’re using process X to achieve Y."
        Consider statements that indirectly signal confusion or confidence. If a user says, "I need that in simpler terms," it implies a lower level of topic familiarity.
        Omit any text that does not directly relate to the topic. Your sole output is the labeled segments that reflect either information about the user or company regarding the topic, OR a message saying "No relevant segments."
        
        This is literature about the topic that you will investigate:
        
        {general_topic_info_summary}
        """)
    

# Define parse prompt for the LLM.
userinfo_parse_system_prompt = f"""You are an advanced parser that transforms a user’s “raw segment” of information into a strict JSON format.

Below are the **theme labels** and their definitions. You must use these to determine which “themes” best match the user’s text. Multiple themes may apply.

{general_framework_info_user}

---

{framework_themes_user}

---

**How to produce your output**:
1. Your output MUST be a valid JSON object with **exactly** these four keys:
   - "topic": A short string describing the main topic (no more than a few words).
   - "score": A numerical value between 0 and 1 (floating point), indicating the perceived level of expertise on the topic.
   - "relevance": A numerical value between 0 and 1 (floating point), indicating how important this information snippet is for assessing the topic.
   - "themes": An array of **one or more** strings containing the relevant theme labels from the above lists.

2. "score" and "relevance" must be numeric between 0 and 1 (for example, 0.0, 0.5, 1.0, etc.).

3. "themes" must be an array of strings—if multiple themes apply, list them all. If no theme clearly applies, you may leave it as an empty array (`"themes": []`).

4. Return only valid JSON with these four keys—no additional keys or text, and no surrounding explanations or markdown."""



companyinfo_parse_system_prompt = f"""You are an advanced parser that transforms a user’s “raw segment” of information into a strict JSON format.

Below are the **theme labels** and their definitions. You must use these to determine which “themes” best match the user’s text. Multiple themes may apply.

{general_framework_info_company}

---

{framework_themes_company}

---

**How to produce your output**:
1. Your output MUST be a valid JSON object with **exactly** these four keys:
   - "description": A short string summarizing the info message (no more than a few words).
   - "score": A numerical value between 0 and 1 (floating point), indicating the perceived level of expertise derived from the info message.
   - "relevance": A numerical value between 0 and 1 (floating point), indicating how important this info message is for the topic.
   - "themes": An array of **one or more** strings containing the relevant theme labels from the above lists.

2. "score" and "relevance" must be numeric between 0 and 1 (for example, 0.0, 0.5, 1.0, etc.).

3. "themes" must be an array of strings—if multiple themes apply, list them all. If no theme clearly applies, you may leave it as an empty array (`"themes": []`).

4. Return only valid JSON with these four keys—no additional keys or text, and no surrounding explanations or markdown."""





async def update_agent_message_history(graph_ctx: GraphRunContext) -> str:
    """
    Constructs and returns a prompt string using the most recent user and frits messages from 
    the internal conversation stored in graph_ctx.deps.conversation_history.
    
    It first searches for the latest 'user' and 'frits' messages by iterating the messages in 
    reverse order. If both are found, they are sorted chronologically and formatted as:
        "<role>: <content>"
    If either message is missing, the entire conversation (sorted by creation time) is used as a fallback.
    """
    # Get all conversation messages from the conversation history.
    messages = list(graph_ctx.deps.conversation_history.values())

    last_user_message = None
    last_frits_message = None

    # Retrieve the most recent user and frits messages.
    for msg in reversed(messages):
        if msg.role.lower() == "user" and last_user_message is None:
            last_user_message = msg
        elif msg.role.lower() == "writer" and last_frits_message is None:
            last_frits_message = msg
        if last_user_message and last_frits_message:
            break

    # If either message is missing, fall back to returning the full conversation history.
    if not last_user_message or not last_frits_message:
        sorted_messages = sorted(messages, key=lambda m: m.created_at)
        return "\n".join(f"{msg.role}: {msg.content}" for msg in sorted_messages)

    # Sort the two messages by their creation time.
    sorted_two = sorted([last_user_message, last_frits_message], key=lambda m: m.created_at)
    return "\n\n".join(f"{msg.role}: {msg.content}" for msg in sorted_two)


def parse_json_result(text: str) -> dict:
    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove markdown code fences if they exist
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        if text.endswith("```"):
            text = text[:-3].strip()

    # First try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract substring between the first '{' and the last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_text = text[start:end+1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ValueError("Failed to extract valid JSON from the result.") from e
        else:
            raise ValueError("No JSON object found in the provided text.")





async def UpdateAgent_workflow(graph_ctx: GraphRunContext) -> GraphRunContext:
    """
    This function does the following in one run:
    
    1. Extraction Phase:
       - It analyzes the last two messages from the conversation (one from the application with role "Frits"
         and one from the user) and extracts raw segments of useful topic information.
       - Each extracted segment is labeled with its category (either "[Company topic Info]" or "[User topic Info]").
       - For each extracted segment, a temporary InfoMessage is created with the raw segment stored in content_str.
       
    2. Parsing Phase:
       - Each temporary InfoMessage is then processed by the same update agent. A prompt is built that asks
         the agent to parse the raw segment (content_str) into a JSON object with keys "topic" and "score".
       - The returned structured JSON is stored in the InfoMessage's content_dict.
       
    3. Finally, each InfoMessage is appended to the appropriate list in the state (new_company_info or new_user_AIR_info)
       based on its category.
    """
    
    update_agent = graph_ctx.deps.update_agent

    ####################################################################
    ####################################################################
    ####################### EXTRACTION PHASE ###########################
    ####################################################################
    ####################################################################
    
    system_prompt_1 = [ModelRequest(parts=[SystemPromptPart(content=extraction_system_prompt)])]

    message_history = await update_agent_message_history(graph_ctx) #### this doesnt give latest two but all messages of the user
      
    # Call the update_agent to extract raw segments.
    extraction_response = await update_agent.run(user_prompt=message_history, message_history=system_prompt_1)
    
    
    ####################################################################
    ####################################################################
    ############## SECONDLY PARSE RAW CONTENT TO DICT ##################
    ####################################################################
    ####################################################################


    system_prompt_2 = [ModelRequest(parts=[SystemPromptPart(content=companyinfo_parse_system_prompt)])] 

    system_prompt_3 = [ModelRequest(parts=[SystemPromptPart(content=userinfo_parse_system_prompt)])]

    if extraction_response:
        lines = extraction_response.data.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            
      ##### COMPANY INFO DISTILLATION PART
            if line.startswith("[Company AIR Info]"):
                segment = line[len("[Company AIR Info]"):].strip()
                result = await update_agent.run(user_prompt=segment, message_history=system_prompt_2)
                logging.debug(f"Parsing result for Company AIR Info: {result}")
            
                try:
                    parsed_data = parse_json_result(result.data)
                except ValueError as e:
                    logging.debug(f"Error parsing Company AIR Info: {e}")
                    parsed_data = {}
                
                company_msg = CompanyInfoMessage(
                    content_str=segment,
                    content_dict=parsed_data
                )
                graph_ctx.state.new_company_info[company_msg.info_id] = company_msg
                logging.debug(f"Appended CompanyInfoMessage: {company_msg}")



      ##### USER INFO DISTILLATION PART
            elif line.startswith("[User AIR Info]"):
                segment = line[len("[User AIR Info]"):].strip()
                result = await update_agent.run(user_prompt=segment, message_history=system_prompt_3)
                logging.debug(f"Parsing result for User AIR Info: {result}")

                try:
                    parsed_data = parse_json_result(result.data)
                except ValueError as e:
                    logging.debug(f"Error parsing User AIR Info: {e}")
                    parsed_data = {}

                user_msg = UserInfoMessage(
                    content_str=segment,
                    content_dict=parsed_data
                )
                graph_ctx.state.new_user_AIR_info[user_msg.info_id] = user_msg
                logging.debug(f"Appended CompanyInfoMessage: {user_msg}")


    return graph_ctx