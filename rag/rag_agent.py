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
    ChatMessage,
    JobProcess
)
from livekit.plugins import (
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import asyncio
import logging
import time
import os

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG manager
_global_rag_manager = None

def prewarm_process(job_process: JobProcess):
    """MANDATORY prewarm - RAG MUST work or process fails"""
    global _global_rag_manager
    
    print("🔥 PREWARM STARTING...")
    logger.info("🔥 PREWARM: Starting RAG initialization...")
    
    try:
        # Ensure we're in the right directory
        current_dir = os.getcwd()
        print(f"🔥 PREWARM: Current directory: {current_dir}")
        
        # Import and initialize
        import asyncio
        import sys
        print(f"🔥 PREWARM: Python path: {sys.path}")
        
        from rag_manager import GenericRAGManager
        print("🔥 PREWARM: Successfully imported GenericRAGManager")
        
        # Check environment variables
        qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        openai_key = os.getenv("OPENAI_API_KEY")
        print(f"🔥 PREWARM: QDRANT_URL: {'SET' if qdrant_url else 'NOT SET'}")
        print(f"🔥 PREWARM: OPENAI_KEY: {'SET' if openai_key else 'NOT SET'}")
        
        async def init_rag():
            global _global_rag_manager
            print("🔥 PREWARM: Creating RAG manager...")
            _global_rag_manager = GenericRAGManager()
            
            print("🔥 PREWARM: Initializing RAG manager...")
            await _global_rag_manager.initialize()
            
            print("🔥 PREWARM: RAG manager initialized successfully!")
            print(f"🔥 PREWARM: RAG manager type: {type(_global_rag_manager)}")
            
            # Test it works
            print("🔥 PREWARM: Testing RAG system...")
            test_result = await _global_rag_manager.search_knowledge("test query")
            print(f"🔥 PREWARM: Test search result: {test_result is not None}")
        
        # Create event loop and run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("🔥 PREWARM: Running RAG initialization...")
        loop.run_until_complete(init_rag())
        loop.close()
        
        # Verify it worked
        if _global_rag_manager is None:
            raise Exception("RAG manager is still None after initialization")
            
        print("✅ PREWARM: RAG system ready!")
        logger.info("✅ PREWARM: Process prewarmed successfully")
        
        # Store in job process userdata as backup
        job_process.userdata["rag_manager"] = _global_rag_manager
        print(f"✅ PREWARM: Stored in userdata: {job_process.userdata.keys()}")
        
    except Exception as e:
        print(f"❌ PREWARM FAILED: {e}")
        logger.error(f"❌ PREWARM FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise  # Fail the prewarm

class Assistant(Agent):
    """RAG-powered voice assistant"""
    
    def __init__(self, rag_manager) -> None:
        super().__init__(
            instructions="""You are an intelligent AI assistant with access to a comprehensive knowledge base.
            
            When users ask questions, you can search through your knowledge base to find relevant information.
            If you find relevant information, incorporate it naturally into your response without explicitly mentioning 
            "knowledge base" or "according to my database". Just provide helpful, accurate information.
            
            If you don't find relevant information in your knowledge base, use your general knowledge to help the user.
            
            If users ask for human support or want to be transferred, offer to transfer them to a human agent.
            
            You can help with any topic that's in your knowledge base. Be conversational, helpful, and accurate.
            Always provide the most relevant and useful information to answer the user's questions."""
        )
        
        self.rag_manager = rag_manager
        self.rag_ready = True
        
        logger.info("✅ Assistant created with working RAG system")

    async def on_user_turn_completed(
        self, 
        turn_ctx: ChatContext, 
        new_message: ChatMessage
    ) -> None:
        """Perform knowledge search for any user query"""
        # Fix: Handle content as list or string
        user_query = new_message.content
        
        # Convert list to string if needed
        if isinstance(user_query, list):
            user_query = " ".join(str(item) for item in user_query if item)
        elif not isinstance(user_query, str):
            user_query = str(user_query) if user_query else ""
        
        if not user_query or len(user_query.strip()) < 5:
            return
            
        # Skip transfer requests
        transfer_keywords = ["human", "person", "agent", "transfer", "representative", "support"]
        if any(keyword in user_query.lower() for keyword in transfer_keywords):
            return
            
        rag_start = time.time()
        logger.info(f"🔍 Knowledge search for: '{user_query[:50]}...'")
        
        try:
            knowledge_context = await self.rag_manager.search_knowledge(user_query)
            
            if knowledge_context:
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
                logger.info("📝 No relevant knowledge found")
                
        except Exception as e:
            rag_time = (time.time() - rag_start) * 1000
            logger.error(f"❌ Knowledge search error after {rag_time:.1f}ms: {e}")

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer function"""
        transfer_to = "sip:voiceai@sip.linphone.org"
        
        job_ctx = get_job_context()
        
        logger.info(f"=== TRANSFER CALL INITIATED ===")
        logger.info(f"Room: {job_ctx.room.name}")
        logger.info(f"Total remote participants: {len(job_ctx.room.remote_participants)}")
        
        sip_participant = None
        for participant in job_ctx.room.remote_participants.values():
            logger.info(f"Found participant: {participant.identity}, kind: {participant.kind}")
            if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                sip_participant = participant
                logger.info(f"✅ Found SIP participant: {participant.identity}")
                break
        
        if not sip_participant:
            logger.error("❌ No SIP participants found!")
            await ctx.session.generate_reply(
                instructions="I'm sorry, I couldn't find any active participants to transfer. Please try calling again."
            )
            return "Could not find any participant to transfer."
        
        participant_identity = sip_participant.identity
        logger.info(f"🔄 Will transfer participant: {participant_identity} to SIP: {transfer_to}")
        
        await ctx.session.generate_reply(
            instructions="""I'm connecting you to a human agent now. The transfer will begin in just a moment. 
            If you hear ringing, the agent should answer automatically. Please stay on the line."""
        )
        
        await asyncio.sleep(2)
        
        try:
            logger.info(f"🚀 Starting SIP transfer request...")
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=participant_identity,
                transfer_to=transfer_to,
                play_dialtone=True,
            )
            
            logger.info(f"📞 Executing transfer_sip_participant...")
            start_time = asyncio.get_event_loop().time()
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=30.0
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            logger.info(f"✅ SIP Transfer completed successfully in {duration:.2f} seconds!")
            return "Call transfer completed successfully to human agent"
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Transfer timeout after 30 seconds")
            await ctx.session.generate_reply(
                instructions="""I'm having trouble connecting to our human agent. The call is reaching them, 
                but their phone isn't automatically answering. Would you like me to try again, or would you prefer to call back later?"""
            )
            return "Transfer timed out - auto-answer not responding."
                    
        except Exception as e:
            logger.error(f"❌ Error transferring call: {e}")
            logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            
            # Provide specific guidance based on error type
            if "408" in str(e):
                logger.error("💡 408 = Call reached destination but timed out waiting for answer")
                await ctx.session.generate_reply(
                    instructions="The call reached our human agent but they didn't answer in time. Please try again or they will call you back shortly."
                )
            elif "500" in str(e):
                logger.error("💡 500 = Server error, possibly SIP configuration issue")
                await ctx.session.generate_reply(
                    instructions="I'm experiencing a technical issue with the transfer system. Please try again in a moment."
                )
            elif "404" in str(e):
                logger.error("💡 404 = SIP address not found")
                await ctx.session.generate_reply(
                    instructions="I couldn't locate our human agent's phone system. Please try again later."
                )
            else:
                await ctx.session.generate_reply(
                    instructions="I apologize, but I'm having trouble transferring your call right now. Please try again in a moment."
                )
            
            return f"Transfer failed: {str(e)}"

    @function_tool()
    async def search_knowledge(self, ctx: RunContext, query: str) -> str:
        """Search the knowledge base for specific information"""
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
- Transferring you to a human agent if needed

Just ask me anything, and I'll search my knowledge base to give you the most accurate information!"""


async def entrypoint(ctx: agents.JobContext):
    """Main entry point"""
    
    logger.info(f"=== AGENT SESSION STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    
    # Try to get RAG manager from global variable first
    global _global_rag_manager
    rag_manager = _global_rag_manager
    
    print(f"🔍 Global RAG manager: {rag_manager}")
    
    # If global doesn't work, try userdata
    if rag_manager is None:
        logger.info("🔍 Trying to get RAG from job process userdata...")
        job_process = ctx.job_process
        if job_process and "rag_manager" in job_process.userdata:
            rag_manager = job_process.userdata["rag_manager"]
            logger.info("✅ Found RAG manager in userdata")
        else:
            logger.error("❌ No RAG manager in userdata either")
    
    # Last resort: initialize RAG now (but this will be slow)
    if rag_manager is None:
        logger.warning("⚠️ No prewarmed RAG found - initializing now (will be slow)")
        try:
            from rag_manager import GenericRAGManager
            rag_manager = GenericRAGManager()
            await rag_manager.initialize()
            logger.info("✅ RAG initialized in entrypoint (slow path)")
        except Exception as e:
            logger.error(f"💀 FATAL: Cannot initialize RAG: {e}")
            raise Exception(f"RAG system failed: {e}")
    
    if rag_manager is None:
        logger.error("💀 FATAL: No RAG system available!")
        raise Exception("RAG system not available")
    
    logger.info("✅ RAG system confirmed available")
    
    # Create session - EXACT SAME as your working telephony agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.3,
        ),
        
        tts=openai.TTS(
            model="tts-1",
            voice="nova",
        ),
        
        vad=silero.VAD.load(),
        
        turn_detection=MultilingualModel(),
    )

    # Create assistant with RAG manager
    assistant = Assistant(rag_manager)

    # Start session - EXACT SAME as your working telephony agent
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Connect immediately
    await ctx.connect()
    logger.info("✅ Agent connected successfully")

    # Send greeting
    await session.generate_reply(
        instructions="""Give a brief, friendly greeting. Say: "Hello! I'm your AI assistant. I can help answer questions using my knowledge base, or transfer you to a human agent if needed. How can I help you today?" Keep it short and professional."""
    )
    
    logger.info("✅ Initial greeting sent")


if __name__ == "__main__":
    logger.info("🚀 Starting RAG Voice Agent with Enhanced Debugging")
    logger.info("📞 Transfer destination: sip:voiceai@sip.linphone.org")
    logger.info("🧠 RAG System: Prewarmed for instant responses")
    
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="my-telephony-agent",
        prewarm_fnc=prewarm_process,
        num_idle_processes=1,
        initialize_process_timeout=60.0,
    ))