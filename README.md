# ClaimCheck

ClaimCheck is a multi-agent AI application that investigates factual claims using evidence from source documents.

## Objective

Build a system that:
- Accepts a claim from a user.
- Retrieves relevant evidence.
- Identifies supporting and contradicting evidence.
- Challenges unsupported conclusions.
- Produces a structured assessment with citations and limitations.

## Planned architecture

- **Research Agent:** Finds relevant sources.
- **Evidence Agent:** Extracts relevant statements.
- **Skeptic Agent:** Challenges weak or contradictory reasoning.
- **Verdict Agent:** Produces the final evidence-backed assessment.

## Technology stack

- Python 3.13
- FastAPI
- Pydantic
- Pytest
- HTTPX
- LangGraph (planned)
- Streamlit (planned)

## Development roadmap

1. Set up the environment and project structure.
2. Build and test the claim-submission API.
3. Create a small evidence collection.
4. Implement the first research and evidence components.
5. Add the skeptic and verdict agents.
6. Introduce multi-agent orchestration.
7. Build a user interface and evaluation suite.

## Current status

Initial environment setup completed. Claim verification functionality is under development.