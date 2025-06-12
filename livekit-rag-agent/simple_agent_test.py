#!/usr/bin/env python3
"""
Simple test of the agent without LiveKit
Tests just the RAG functionality
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_rag():
    """Test the agent RAG functionality without LiveKit"""
    try:
        from scalable_fast_rag import scalable_rag
        
        logger.info("🤖 Testing Agent RAG functionality...")
        
        # Initialize RAG
        success = await scalable_rag.initialize()
        if not success:
            logger.error("❌ RAG initialization failed")
            return
        
        # Simulate user queries
        user_queries = [
            "What services do you provide?",
            "Tell me about your pricing",
            "What are your business hours?", 
            "How can I contact support?",
            "What features do you offer?",
            "Can you integrate with existing systems?",
            "Do you support multiple languages?",
            "What is your uptime guarantee?"
        ]
        
        logger.info("🎯 Simulating user conversations...")
        
        for query in user_queries:
            logger.info(f"\n👤 User: {query}")
            
            # Simulate the agent's context injection process
            try:
                # This is what happens in on_user_turn_completed
                results = await asyncio.wait_for(
                    scalable_rag.quick_search(query),
                    timeout=0.15  # Same timeout as agent
                )
                
                if results and len(results) > 0:
                    context = results[0]["content"]
                    logger.info(f"🧠 Agent Context: {context[:150]}...")
                    logger.info(f"🤖 Agent Response: Based on our knowledge base - {context[:100]}...")
                else:
                    logger.info(f"🤖 Agent Response: I don't have specific information about that. Would you like me to transfer you to a human agent?")
                    
            except asyncio.TimeoutError:
                logger.info(f"🤖 Agent Response: Let me connect you with a human agent for assistance.")
            except Exception as e:
                logger.error(f"❌ Error processing query: {e}")
                logger.info(f"🤖 Agent Response: I'm having trouble accessing information. Let me transfer you to an agent.")
        
        logger.info("\n✅ Agent RAG test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_rag())