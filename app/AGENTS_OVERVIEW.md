# Frits Back-end Agents Overview

This document explains the roles and step-by-step flows of the main AI agents in the Frits back-end. It is intended for both developers and non-technical team members.

---

## Meta Agent
**Purpose:** Analyzes interview data and creates a comprehensive interview context for the AI system.

**Steps:**
1. Receives system knowledge, topic-specific info, and dynamic user information (e.g., last session, user profile, feedback).
2. Distills this information into an “interview context.”
3. Updates the context if feedback from the Reviewer Agent is present.
4. Passes the context to the next agent (usually the Writer Agent) for further action.

---

## Reviewer Agent
**Purpose:** Evaluates the interview context produced by the Meta Agent for gaps, inconsistencies, or missing information.

**Steps:**
1. Receives the draft interview context from the Meta Agent.
2. Reads and analyzes the context, comparing it to conversation logs and user profiles.
3. Identifies improvements, clarifications, or missing details.
4. May call a Retrieval-Augmented Generation (RAG) tool for additional authoritative data if needed.
5. Provides feedback (and/or approval) to improve the interview context.
6. Checks if the session is finished or needs further review.

---

## Writer Agent
**Purpose:** Drives the interview by generating the next step or response.

**Steps:**
1. Receives the interview context, user sentiment, interview stage, and profile.
2. Determines the best next action (e.g., ask a question, offer an explanation).
3. Focuses on uncovering gaps in AI readiness, adapting style to the user’s responses.
4. Avoids overloading the user; asks one question at a time and rotates topics if needed.
5. Provides responses in a conversational style suitable for speech.

---

## Update Agent
**Purpose:** Updates user or company information related to AI readiness.

**Steps:**
1. Receives messages about company or user info.
2. Uses a framework of AI readiness dimensions (e.g., technology knowledge, experience).
3. Processes updates to scores or dimensions based on new data.
4. Ensures the backend’s understanding of AI readiness is current and accurate.
5. May clarify what represents good or bad scores for each dimension.

---

*For more details or technical questions, contact a developer. For business or feature questions, reach out to the product owner or business analyst.*