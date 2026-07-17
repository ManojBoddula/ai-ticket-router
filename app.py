import os
from typing import List, Dict

# =====================================================================
# MODULE 1: MODERN DATA INGESTION CORE
# Using the updated, long-term supported LangChain packages to clear warnings.
# =====================================================================
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class TicketAutomationSystem:
    def __init__(self):
        # =============================================================
        # MODULE 2: INITIALIZING THE RE-ENGINEERED ML ENGINE
        # Loading the vector embeddings and initializing our Chroma store.
        # =============================================================
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(embedding_function=self.embeddings)
        
        # Seed our system with corporate knowledge right away
        self._seed_knowledge_base()

    def _seed_knowledge_base(self):
        # =============================================================
        # MODULE 3: PRE-LOADING PAST RESOLUTION HISTORY
        # Seeding the database with clear intent maps and verified fixes.
        # =============================================================
        historical_data = [
            Document(
                page_content="I forgot my password. How do I change it?",
                metadata={
                    "intent": "auth_reset", 
                    "solution": "Click 'Forgot Password' on the login screen, type your corporate email, and follow the reset link sent to your inbox."
                }
            ),
            Document(
                page_content="How can I check my remaining vacation or annual leave days?",
                metadata={
                    "intent": "hr_leave", 
                    "solution": "Log into the EmployWise Portal -> Employee Self Service -> Leave Balance Tab to check your balance."
                }
            )
        ]
        self.vector_store.add_documents(historical_data)

    def process_ticket(self, ticket_text: str, similarity_threshold: float = 0.35) -> Dict:
        # =============================================================
        # MODULE 4: SEMANTIC SEARCHING
        # We query ChromaDB using similarity search to find the closest intent.
        # =============================================================
        results = self.vector_store.similarity_search_with_relevance_scores(ticket_text, k=1)
        
        # =============================================================
        # MODULE 5: THE ANTI-HALLUCINATION GUARDRAIL
        # Adjusted threshold to align cleanly with the distance metrics 
        # returned by the lightweight local transformer model.
        # =============================================================
        if not results or results[0][1] < similarity_threshold:
            return {
                "ticket": ticket_text,
                "action": "ROUTE_TO_HUMAN",
                "confidence": results[0][1] if results else 0.0,
                "response": "I'm passing this request over to a human support agent to make sure it's answered correctly."
            }
            
        # Unpack the valid match if the confidence score passes the threshold
        matched_doc, score = results[0]
        
        # =============================================================
        # MODULE 6: GENERATING THE RESOLUTION
        # Returns the automated fix matching the group intent.
        # =============================================================
        return {
            "ticket": ticket_text,
            "action": f"AUTO_RESOLVE_{matched_doc.metadata['intent'].upper()}",
            "confidence": round(score, 3),
            "response": matched_doc.metadata['solution']
        }

# =====================================================================
# SYSTEM RUNTIME LOOP (Quick local test verification)
# =====================================================================
if __name__ == "__main__":
    # Spin up the AI ticketing agent
    ai_agent = TicketAutomationSystem()
    
    # Simulating the raw user text submissions
    test_tickets = [
        "I can't log in, as password is incorrect.",
        "How to see leave balance?"
    ]
    
    # Process both streams through our modules and see how they route
    for ticket in test_tickets:
        resolution = ai_agent.process_ticket(ticket)
        print(f"\n[Incoming Ticket]: {resolution['ticket']}")
        print(f"[Action Taken]: {resolution['action']} (Confidence: {resolution['confidence']})")
        print(f"[AI Automated Output]: {resolution['response']}")