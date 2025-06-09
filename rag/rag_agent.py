from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import (
    Agent, 
    AgentSession, 
    RoomInputOptions, 
    RunContext,
    function_tool,
    get_job_context,
    ChatContext,
    ChatMessage
)
from livekit.plugins import (
    openai,
    elevenlabs,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import asyncio
import logging
import time

from rag_manager import GenericRAGManager

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericRAGAssistant(Agent):
    """Generic RAG-powered voice assistant that works with any knowledge domain"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an intelligent AI assistant with access to a comprehensive knowledge base.
            
            When users ask questions, you can search through your knowledge base to find relevant information.
            If you find relevant information, incorporate it naturally into your response without explicitly mentioning 
            "knowledge base" or "according to my database". Just provide helpful, accurate information.
            
            If you don't find relevant information in your knowledge base, use your general knowledge to help the user.
            
            You can help with any topic that's in your knowledge base. Be conversational, helpful, and accurate.
            Always provide the most relevant and useful information to answer the user's questions."""
        )
        
        # Initialize RAG system
        self.rag_manager = GenericRAGManager()
        self.rag_ready = False
        
    async def initialize_rag(self):
        """Initialize RAG system"""
        try:
            await self.rag_manager.initialize()
            self.rag_ready = True
            logger.info("✅ Generic RAG system ready")
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            self.rag_ready = False

    async def on_user_turn_completed(
        self, 
        turn_ctx: ChatContext, 
        new_message: ChatMessage
    ) -> None:
        """Perform knowledge search for any user query"""
        if not self.rag_ready:
            logger.info("📝 RAG not ready, using general knowledge")
            return
            
        user_query = new_message.content
        if not user_query or len(user_query.strip()) < 5:
            return  # Skip very short queries
            
        rag_start = time.time()
        logger.info(f"🔍 Knowledge search for: '{user_query[:50]}...'")
        
        try:
            # Search knowledge base
            knowledge_context = await self.rag_manager.search_knowledge(user_query)
            
            if knowledge_context:
                # Add context to chat
                enhanced_context = f"""[RELEVANT INFORMATION FROM KNOWLEDGE BASE]
{knowledge_context}

[INSTRUCTIONS: Use the above information to enhance your response if it's relevant to the user's question. 
Incorporate the information naturally into your answer. If the information isn't relevant, ignore it and respond normally.]"""
                
                turn_ctx.add_message(
                    role="system",
                    content=enhanced_context
                )
                
                rag_time = (time.time() - rag_start) * 1000
                logger.info(f"✅ Knowledge context added: {rag_time:.1f}ms")
            else:
                logger.info("📝 No relevant knowledge found, using general knowledge")
                
        except Exception as e:
            rag_time = (time.time() - rag_start) * 1000
            logger.error(f"❌ Knowledge search error after {rag_time:.1f}ms: {e}")

    @function_tool()
    async def search_knowledge(self, ctx: RunContext, query: str) -> str:
        """Search the knowledge base for specific information"""
        if not self.rag_ready:
            return "Knowledge search is not available at the moment. I'll help you with what I know from my general training."
            
        try:
            # Provide user feedback
            await ctx.session.generate_reply(
                instructions=f'Let the user know you\'re searching for information about "{query}" in your knowledge base.'
            )
            
            result = await self.rag_manager.search_knowledge(query)
            
            if result:
                return f"Here's what I found in my knowledge base:\n\n{result}"
            else:
                return "I couldn't find specific information about that in my current knowledge base. Let me help you with what I know."
                
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "I encountered an error while searching. Let me help you with my general knowledge instead."

    @function_tool()
    async def get_knowledge_stats(self, ctx: RunContext) -> str:
        """Get information about the knowledge base"""
        if not self.rag_ready:
            return "Knowledge base is not currently available."
            
        try:
            collection_info = await self.rag_manager.get_collection_info()
            performance_stats = self.rag_manager.get_performance_stats()
            
            return f"""Knowledge Base Information:
Collection: {collection_info.get('collection_name', 'Unknown')}
Documents: {collection_info.get('vectors_count', 'Unknown')} entries
Performance: {performance_stats.get('avg_latency_ms', 'N/A')}ms average search time
Cache Hit Rate: {performance_stats.get('cache_hit_rate', 'N/A')}"""
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return "Unable to retrieve knowledge base statistics at this time."

    @function_tool()
    async def help_commands(self, ctx: RunContext) -> str:
        """Show available commands and capabilities"""
        return """I'm an AI assistant with access to a knowledge base. I can help you by:

- Answering questions using my knowledge base
- Searching for specific information with 'search_knowledge'
- Providing general assistance on any topic
- Showing knowledge base statistics with 'get_knowledge_stats'

Just ask me anything, and I'll search my knowledge base to give you the most accurate information!"""


async def entrypoint(ctx: agents.JobContext):
    """Main entry point for the generic RAG voice agent"""
    
    logger.info(f"=== GENERIC RAG AGENT STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    logger.info(f"Agent: generic-rag-agent")
    
    # Create the session
    session = AgentSession(
        # STT: Speech-to-text
        stt=deepgram.STT(model="nova-3", language="multi"),
        
        # LLM: Language model
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=250,
        ),
        
        # TTS: Text-to-speech
        tts=openai.TTS(
            model="tts-1",
            voice="nova",
        ),
        
        # VAD: Voice activity detection
        vad=silero.VAD.load(),
        
        # Turn detection
        turn_detection=MultilingualModel(),
    )

    # Create assistant
    assistant = GenericRAGAssistant()
    
    # Initialize RAG system
    await assistant.initialize_rag()

    # Start the session
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Connect to the room
    await ctx.connect()
    logger.info("✅ Generic RAG agent connected successfully")

    # Generate initial greeting
    await session.generate_reply(
        instructions="""Give a brief, friendly greeting. Say something like: "Hello! I'm your AI assistant with access to a knowledge base. I can help answer questions on various topics using the information I have available. What would you like to know?" Keep it welcoming and open-ended."""
    )
    
    logger.info("✅ Initial greeting sent")


if __name__ == "__main__":
    logger.info("🚀 Starting Generic RAG Voice Agent")
    logger.info("🧠 RAG System: Universal Knowledge Base Assistant")
    
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="generic-rag-agent"
    ))