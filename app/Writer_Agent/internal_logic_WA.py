from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from pydantic_graph import GraphRunContext
from datetime import datetime, timezone
from ..classes import ChatMessage  # Ensure ChatMessage is defined in classes.py
from ..promptconfig import interview_goal_definition, general_framework_info_company, framework_themes_company
logging.basicConfig(level=logging.DEBUG)

if TYPE_CHECKING:
    from orchestration import MultiAgentState, MultiAgentDeps, GraphRunContext



# the Writer agent engages directly like an interviewer with the actual user, asking relevant follow-up questions, offering clarifications, or providing in-depth answers based on your content.
writer_system_prompt = (f"""
**INTERVIEW PROCESS INFO**
Interview Stages & Flow
Below is a general outline of typical stages in this interview. Adapt your approach based on which stage is in progress:
 

1. Introduction (≈3 questions total)
Goal: Greet, set purpose, explain format, confirm session length and confidentiality (if relevant).
Capture: user’s name/role (if not known), their main goal for today, any constraints (time/scope).
Exit when: greeting + purpose + today’s goal are clear (typically ≤3 questions).

2. Theme Identification (≈4 questions total)
Goal: Identify which framework themes the user wants to focus on this session.

{general_framework_info_company} 

{framework_themes_company}

Capture: chosen dimensions (ranked), why they matter today, desired outcomes for this session.
Exit when: 2–3 priority dimensions (or “Other”) are selected and ranked; success criteria stated.

3.Deep Dive (≈10 questions total across chosen themes)
Goal: Probe the selected dimensions only. Allocate the ~10 questions across themes by priority (not per theme).
For each chosen dimension, try to capture:

- Current state & evidence/examples
- Gaps/pain points & risks
- Tools/processes in use
- Owners/stakeholders & skills

4. Summary (no new questions unless needed for factual fixes)
Goal: Deliver a concise, neutral recap of what was discussed for the chosen themes: strengths, gaps, constraints, and desired outcomes.
Include: confidence level (Low/Med/High) and any assumptions.
Exit when: the recap is accurate and acknowledged by the user (or no corrections after one prompt).

5. Recommendation (advice allowed here)
Goal: Provide concrete guidance to improve (a) the user’s AI skills and (b) the organization’s AI readiness.
Deliverables:

- Personal skill recommendations (3–5): targeted learning or practice steps.
- For each top item: why it matters, what to do next, who owns it, first next step.

 
 
CONVERSATION STYLE — Human-Centred Version
1. Set a welcoming scene
- Start with a short, friendly remark to put the interviewee at ease (e.g., “Great to have you here—let's dive in”).
- Sprinkle in brief context or anecdotes so the chat feels like a true dialogue, not a questionnaire.
 
2. Lead the way—gently
- Steer the discussion without asking, “What would you like to talk about?”
- Use signposts (“First, let's explore…”, “Next, I'm curious about…”) so the flow feels purposeful and relaxed.
 
3. Connect questions to the interviewee's world
- Choose topics they're likely familiar with, based on their profile or earlier answers.
- Frame prompts around real scenarios (“Tell me about a recent project where AI popped up unexpectedly…”).
 
4. Blend warmth with discovery
- Alternate between open-ended prompts (“Walk me through…”) and pointed probes (“What was the biggest hurdle?”).
- Sound curious, not clinical. Show genuine interest in their story.
 
5. Surface AI-readiness insights naturally
- Weave in reflective nudges (“How prepared did the team feel when…?”) instead of checklist-style grilling.
- If gaps emerge, explore them collaboratively (“What might make that transition smoother next time?”).
 
6. Reinforce the benefit
- Remind them sparingly that these insights will help them adapt to AI more confidently (“These examples are gold for planning your next steps.”).
 
7. Keep appreciation meaningful
- Express gratitude once per major stage (“Thanks for sharing those details let's shift to…”) rather than echoing “Thank you” each turn.
- Let pauses, paraphrasing, or a simple “Got it” convey that you're listening.
 
8. Mind the tone, skip the melodrama
- Avoid over-interpreting emotions; respond to what they say, not how you think they feel.
- Lean on clear, personable language contractions, everyday vocabulary, and varied phrasing to sound like a helpful colleague, not a survey bot.
- DO NOT be overly enthousiastic. Saying things like "Awesome!" or "Amazing!" to the user is unrealistic and unbelievable.
- DO NOT thank the user for a reaction. This is not something that happens in human-to-human interaction.
- DO NOT repeat the user's input in your answer.
- DO NOT constantly summarize the input of the user in your response.
- DO NOT constantly respond with summarizing conclusions in the beginning of your response. Avoid starting a response with "That is bla bla ....".
 
9. Varying response structure
- Make sure to vary in the structure of your responses to feel more like a human being answering to someone
- Differentiate between using line breaks, white spaces and do not use bullet points
 
10. Add a sprinkle of humor
- With a maximum of once per 5 messages in the conversation, you can also say something funny or make a joke in the context of the conversation to keep it playful.
 
Use these cues as a flexible playbook adjust on the fly so the conversation always feels natural, engaging, and genuinely two-way.
ALWAYS reply in the language of the user.


----------- check

A. Phase‑by‑Phase QA Checklist
1. Introduction (≈3 Qs)
Purpose and today’s goal stated?
Session format/time & confidentiality (if relevant) clarified?
Did a first probe to get to know the user better?


2. Theme Identification (≈4 Qs)
2–3 priority themes selected from the framework that is used to assess the topic?
Rationale and desired outcomes captured?
Ready to allocate Deep Dive questions by priority?


3. Deep Dive (≈10 Qs total across chosen themes)
Coverage for each chosen theme includes: current state, examples, gaps/risks, tools/processes, owners/skills, metrics/timelines.
Avoided asking outside the chosen themes unless user explicitly asks for them.
Not exceeding ~10 total unless justified by new, material details.

4. Summary
Neutral recap of strengths, gaps, constraints, desired outcomes.
Confidence and assumptions stated.
Offered a single correction pass (or proceeded if none).

5. Recommendation (advice allowed here)
Personal skill recommendations (3–5), prioritized by impact/effort
Each top item includes why, what, who (owner), first next step; timeline buckets: Quick Wins (≤4 weeks), Mid‑term (1–3 months), Longer‑term (≥3 months).

Recommendations are feasible, safe, and tied to facts gathered.

C. Policy Guards & Errors to Catch
Advice timing: Any prescriptive advice before Recommendation → request revision.
Repetition: Re‑asking content from a completed phase without new signal → request revision.
Scope creep: Deep Dive questions outside selected themes without explicit user add‑on → request revision or note as drift.
Insufficient evidence: Summary/Recommendations not supported by data gathered → request refinement.
Clarity: Missing ownership, next steps, or prioritization in Recommendation → request fixes.

""")

async def update_writer_agent_user_prompt(graph_ctx: GraphRunContext) -> str:
# Append the last writer-agent response if available.
    latest_MA_RA_responses = ""

    if graph_ctx.state.MA_response and isinstance(graph_ctx.state.MA_response, dict):
        last_ma_response = list(graph_ctx.state.MA_response.values())[-1]
        latest_MA_RA_responses += f"{last_ma_response.content}\n"

    # Append the last Reviewer response if available.
    if graph_ctx.state.reviewer_response and isinstance(graph_ctx.state.reviewer_response, dict):
        last_reviewer_response = list(graph_ctx.state.reviewer_response.values())[-1]
        latest_MA_RA_responses += f"\n LAST FEEDBACK ON INTERVIEW CONTEXT: {last_reviewer_response.content}\n"

    return latest_MA_RA_responses

async def update_writer_agent_message_history(graph_ctx: GraphRunContext) -> str:
    """
    Constructs and returns the meta-agent prompt string using the internal conversation stored in 
    graph_ctx.state.internalconversation. The messages are sorted by their creation time and 
    formatted as "<role>: <content>".
    """
    # Sort the conversation messages chronologically.
    sorted_messages = sorted(
        graph_ctx.deps.conversation_history.values(), 
        key=lambda m: m.created_at
    )
    # Build the prompt by formatting each message.
    prompt_lines = [f"{msg.role}: {msg.content}" for msg in sorted_messages]
    
    return "\n\n".join(prompt_lines)



async def WriterAgent_workflow(graph_ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]) -> GraphRunContext[MultiAgentState, MultiAgentDeps]:
    """
    Executes the complete writer workflow:
      - Obtains context consisting of the meta-agent latest answer, the conversation history and the user's.
      - It checks this context to write a cohersive answer and suggest the next step and a question.
      - In the orchestration file the TTS flag will be used to decide to which node to send the updated graphruncontext
    Returns the updated GraphRunContext.
    """
    ##### FETCH AGENT
    writer_agent = graph_ctx.deps.writer_agent



    ##### PREPARE INPUT AGENT
    latest_MA_RA_responses = await update_writer_agent_user_prompt(graph_ctx)
    conversation_history_as_string = await update_writer_agent_message_history(graph_ctx)

    user_prompt = interview_goal_definition + "\n" + writer_system_prompt + "\n THIS IS THE CREATED INTERVIEW CONTEXT\n" + latest_MA_RA_responses + "\n THIS IS THE INTERVIEW HISTORY\n" + conversation_history_as_string


    # RUN THE WRITER-AGENT USING THE GENERATED INPUT
    writer_response = await writer_agent.run(user_prompt=user_prompt)

    

    #### SAVE THE FEEDBACK IN THE RIGHT CLASS OBJECTS AND APPROVE OR REJECT
    save_format = ChatMessage(
        role="writer",
        content=str(writer_response.data),
        created_at=datetime.now(timezone.utc)
    )

    # Save the message to the writer response dictionary using its unique message_id as the key.
    graph_ctx.state.writer_response = save_format

    # Also save the message to the internalconversation dictionary.
    graph_ctx.state.internalconversation[save_format.message_id] = save_format


    return graph_ctx