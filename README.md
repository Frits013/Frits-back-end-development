# Frits Back-end (development)

## Overview
This folder contains the backend code for the Frits platform. It powers the application's core logic, agent-based automation, and API endpoints. The backend is built with Python and is designed for modularity, scalability, and ease of collaboration between developers and non-technical team members.

---

## What Does This Backend Do?
- **Handles requests from the frontend** (e.g., chat, authentication, content generation)
- **Coordinates AI agents** to perform tasks like writing, reviewing, updating, and orchestrating workflows
- **Exposes RESTful APIs** for communication with other systems

---

## Folder Structure

- `requirements.txt` — Lists Python packages needed to run the backend
- `app/` — Main backend application
  - `main.py` — Entry point for starting the backend server
  - `auth.py` — Authentication logic (login, signup, etc.)
  - `classes.py` — Core data models and classes
  - `dependencies.py` — Shared resources and dependency management
  - `orchestration.py` — Coordinates workflows and agent interactions
  - `routes/` — API endpoints
    - `auth_routes.py` — Authentication endpoints
    - `chat_routes.py` — Chat-related endpoints
  - `Meta_Agent/` — High-level coordination agent
  - `Reviewer_Agent/` — Agent for reviewing content
  - `Update_Agent/` — Agent for updating data or models
  - `Writer_Agent/` — Agent for generating content

---

## How Does It Work?
1. **A user or system sends a request** (e.g., to start a chat or generate content)
2. **The request is routed** to the correct API handler
3. **Agents process the request** (e.g., Writer Agent generates text, Reviewer Agent checks it)
4. **A response is sent back** to the requester

---

## For Developers
- Modular agent design makes it easy to add new features or agents
- API routes are organized for clarity and maintainability
- Shared logic is separated for reuse and testing
- See `main.py` for how the server is started and agents are wired together

---

## For Stakeholders & Business Analysts
- Each agent represents a business function (e.g., writing, reviewing, updating)
- The backend is responsible for automating and coordinating these functions
- API endpoints allow integration with other systems or user interfaces
- The structure supports future growth and new business requirements

---

## Getting Started
1. **Install Python dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Run the backend server:**
   ```powershell
   python app/main.py
   ```
3. **Access API endpoints** via the frontend or tools like Postman

---

## Need Help?
- For technical questions, contact a developer
- For business or feature questions, reach out to the product owner or business analyst

---

*This README is intended for both technical and non-technical team members. It explains the purpose, structure, and usage of the Frits backend in clear terms.*