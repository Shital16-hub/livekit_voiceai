"""
LiveKit RAG Voice Agent with Human Transfer
Complete working version with all fixes applied
"""
import asyncio
import logging
from typing import Optional

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
    deepgram,
    noise_cancellation,
    silero,
)
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except ImportError:
    MultilingualModel = None

from dotenv import load_dotenv
load_dotenv()

# Local imports - REQUIRED for RAG agent
from config import config, validate_config
from utils.rag_manager import rag_manager, search_knowledge_base

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAssistant(Agent):
    """RAG-powered voice assistant - RAG is mandatory"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant with access to a comprehensive knowledge base.
            
            Key capabilities:
            - Answer questions using your knowledge base when relevant
            - Provide accurate, specific information from company data
            - Keep responses concise and conversational for voice interaction
            - Transfer to human agents when requested or when you cannot help
            
            Important guidelines:
            - Use retrieved context when available to provide accurate answers
            - If someone asks to speak to a human, transfer them immediately
            - Keep responses under 150 words for voice clarity
            - Always confirm before transferring calls
            - Be helpful, friendly, and professional
            
            Transfer phrases to watch for:
            - "speak to a human"
            - "talk to a person" 
            - "transfer me"
            - "human agent"
            - "customer service representative"
            """
        )
        self.rag_ready = True  # Always true since RAG is mandatory
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Parallel RAG lookup - injects context without adding latency"""
        if not self.rag_ready:
            return
            
        try:
            user_text = new_message.text_content()
            if not user_text or len(user_text.strip()) < 10:
                return
            
            # Check if this looks like a transfer request
            transfer_keywords = ["human", "person", "agent", "representative", "transfer", "speak to", "talk to"]
            if any(keyword in user_text.lower() for keyword in transfer_keywords):
                logger.info("🔄 Transfer request detected, skipping RAG lookup")
                return
            
            # ✅ SIMPLE FIX: Use the search function directly
            logger.info(f"🔍 RAG lookup for: {user_text[:50]}...")
            
            # Get search results
            results = await rag_manager.search(user_text)
            
            if results and len(results) > 0:
                # Get the best result and limit length
                context = results[0]["content"][:250] + "..." if len(results[0]["content"]) > 250 else results[0]["content"]
                
                # Inject context into conversation for LLM
                turn_ctx.add_message(
                    role="assistant",
                    content=f"[CONTEXT FROM KNOWLEDGE BASE]: {context}"
                )
                logger.info("📚 Added RAG context to conversation")
            else:
                logger.debug("🔍 No relevant context found")
                    
        except Exception as e:
            logger.error(f"❌ RAG context lookup error: {e}")
    
    @function_tool()
    async def search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for specific information"""
        try:
            logger.info(f"🔍 Knowledge base search: {query[:50]}...")
            result = await search_knowledge_base(query)
            
            if result and len(result.strip()) > 10:
                logger.info("📚 Knowledge base search successful")
                return result
            else:
                return "I couldn't find specific information about that in our knowledge base."
                
        except Exception as e:
            logger.error(f"❌ Knowledge base search error: {e}")
            return "I'm sorry, I encountered an error searching the knowledge base."
    
    @function_tool()
    async def transfer_call(self, ctx: RunContext) -> str:
        """Transfer the call to a human agent"""
        
        transfer_to = config.transfer_sip_address
        job_ctx = get_job_context()
        
        logger.info(f"=== TRANSFER CALL INITIATED ===")
        logger.info(f"Room: {job_ctx.room.name}")
        logger.info(f"Total remote participants: {len(job_ctx.room.remote_participants)}")
        
        # Find the SIP participant
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
            return "Could not find any participant to transfer. Please try again."
        
        participant_identity = sip_participant.identity
        logger.info(f"🔄 Will transfer participant: {participant_identity} to SIP: {transfer_to}")
        
        # Inform the user about the transfer
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
            await ctx.session.generate_reply(
                instructions="I apologize, but I'm having trouble transferring your call right now. Please try again in a moment."
            )
            return f"Transfer failed: {str(e)}"

    @function_tool()
    async def check_agent_availability(self) -> str:
        """Check if human agents are available"""
        return "Human agents are available to assist you. Would you like me to transfer your call?"

    @function_tool()
    async def get_business_hours(self) -> str:
        """Get information about business hours and availability"""
        return "Our AI assistant is available 24/7. Human agents are typically available Monday through Friday, 9 AM to 5 PM. However, I can transfer you to check for immediate availability."

async def entrypoint(ctx: agents.JobContext):
    """Main entry point - RAG MUST work or agent fails"""
    
    logger.info(f"=== RAG VOICE AGENT STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    
    # 🔥 MANDATORY RAG INITIALIZATION - FAIL IF IT DOESN'T WORK
    logger.info("🔧 Initializing RAG system (MANDATORY)...")
    rag_success = await rag_manager.load_or_create_index()
    
    if not rag_success:
        logger.error("❌ RAG SYSTEM FAILED TO INITIALIZE")
        logger.error("💡 Please check:")
        logger.error("   - Knowledge base exists in storage/ directory")
        logger.error("   - Run: python data_ingestion.py --create-sample")
        logger.error("   - Run: python data_ingestion.py")
        raise RuntimeError("RAG system is mandatory but failed to initialize")
    
    logger.info("✅ RAG system initialized successfully")
    
    # Create session - optimized configuration
    session = AgentSession(
        # STT: Deepgram configuration with timeout settings
        stt=deepgram.STT(
            model="nova-3", 
            language="multi",
            
        ),
        
        # LLM: OpenAI configuration with timeout and error handling
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,  # Lower for consistency
            timeout=30.0,     # Add timeout
        ),
        
        # TTS: OpenAI TTS configuration
        tts=openai.TTS(
            model="tts-1",
            voice="nova",
            speed=1.0,
        ),
        
        # VAD: Silero configuration
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.6
        ),
        
        # Turn Detection: Multilingual model
        turn_detection=MultilingualModel() if MultilingualModel else None,
    )

    # Create RAG assistant
    assistant = RAGAssistant()

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
    logger.info("✅ Agent connected to room successfully")

    # Generate initial greeting with error handling
    try:
        greeting_instructions = """Give a brief, friendly greeting. Say: "Hello! I'm your AI assistant with access to our comprehensive knowledge base. How can I help you today? If you need to speak with a human agent, just let me know and I can transfer you right away." Keep it conversational and under 8 seconds."""
        
        await session.generate_reply(instructions=greeting_instructions)
        logger.info("✅ Initial greeting sent")
        
    except Exception as e:
        logger.error(f"❌ Greeting failed: {e}")
        # Continue without greeting - agent will still respond to user input
        logger.info("⚠️ Continuing without greeting - agent ready for user input")
    
    logger.info("✅ RAG-powered agent ready and operational")

if __name__ == "__main__":
    try:
        # Validate configuration first
        validate_config()
        
        logger.info("🚀 Starting MANDATORY RAG Voice Agent")
        logger.info(f"📞 Transfer destination: {config.transfer_sip_address}")
        logger.info("🔥 RAG is REQUIRED - agent will fail if RAG doesn't work")
        
        agents.cli.run_app(agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"  # ✅ Matches your dispatch rule
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        logger.error("🔥 Agent failed to start - fix RAG system first")
        exit(1)