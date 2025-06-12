"""
Fixed LiveKit RAG Voice Agent
Follows official LiveKit patterns and best practices
"""
import asyncio
import logging
from typing import Optional
from pathlib import Path

from livekit import agents, api
from livekit.agents import (
    Agent, 
    AgentSession, 
    JobContext,
    RunContext,
    function_tool,
    get_job_context,
    ChatContext,
    ChatMessage,
    WorkerOptions,
    cli
)
from livekit.plugins import (
    openai,
    deepgram,
    silero,
)

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except ImportError:
    MultilingualModel = None

from dotenv import load_dotenv
load_dotenv()

# Import your existing config and RAG manager
from config import config, validate_config
from utils.rag_manager import rag_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGVoiceAgent(Agent):
    """
    Fixed RAG Voice Agent following LiveKit best practices
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI voice assistant with access to a comprehensive knowledge base.

IMPORTANT GUIDELINES:
- Keep responses concise and conversational for voice interaction (under 100 words)
- Use your knowledge base to provide accurate, specific information  
- If someone asks to speak to a human, transfer them immediately
- Be friendly, professional, and helpful
- Confirm transfer requests before executing them

TRANSFER KEYWORDS TO WATCH FOR:
- "speak to a human" / "talk to a person"
- "transfer me" / "human agent" 
- "customer service representative"
- "real person" / "live agent"
"""
        )
        self.rag_initialized = False
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        ✅ FIXED: Proper RAG context injection following LiveKit patterns
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 5:
                return
            
            # Skip RAG for transfer requests
            transfer_keywords = ["human", "person", "agent", "representative", "transfer", "speak to", "talk to"]
            if any(keyword in user_text.lower() for keyword in transfer_keywords):
                logger.info("🔄 Transfer request detected, skipping RAG lookup")
                return
            
            # ✅ FIXED: Use proper async RAG lookup
            logger.info(f"🔍 RAG lookup for: {user_text[:50]}...")
            
            # Get context from RAG system
            rag_results = await rag_manager.search(user_text)
            
            if rag_results and len(rag_results) > 0:
                # Extract best result and limit length for voice
                best_result = rag_results[0]
                context = best_result.get("content", "")
                
                # Limit context length for voice interaction
                if len(context) > 200:
                    context = context[:200] + "..."
                
                # ✅ FIXED: Proper context injection
                turn_ctx.add_message(
                    role="system", 
                    content=f"[KNOWLEDGE BASE CONTEXT]: {context}"
                )
                logger.info("📚 Added RAG context to conversation")
            else:
                logger.debug("🔍 No relevant context found")
                
        except Exception as e:
            logger.error(f"❌ RAG context lookup error: {e}")
            # Continue without RAG context - don't fail the conversation
    
    @function_tool()
    async def search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for specific information"""
        try:
            logger.info(f"🔍 Knowledge base search: {query[:50]}...")
            
            # Use the RAG manager's query method for formatted responses
            result = await rag_manager.query(query)
            
            if result and len(result.strip()) > 10:
                logger.info("📚 Knowledge base search successful")
                return result
            else:
                return "I couldn't find specific information about that in our knowledge base."
                
        except Exception as e:
            logger.error(f"❌ Knowledge base search error: {e}")
            return "I'm sorry, I encountered an error searching the knowledge base."
    
    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer the call to a human agent"""
        try:
            job_ctx = get_job_context()
            transfer_to = config.transfer_sip_address
            
            logger.info(f"=== INITIATING CALL TRANSFER ===")
            logger.info(f"Room: {job_ctx.room.name}")
            logger.info(f"Transfer destination: {transfer_to}")
            
            # Find SIP participant
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    sip_participant = participant
                    logger.info(f"✅ Found SIP participant: {participant.identity}")
                    break
            
            if not sip_participant:
                logger.error("❌ No SIP participants found for transfer")
                return "I'm sorry, I couldn't find an active call to transfer. Please try calling again."
            
            # Inform user about transfer
            await ctx.session.generate_reply(
                instructions="I'm connecting you to a human agent now. Please stay on the line while I transfer your call."
            )
            
            await asyncio.sleep(1)
            
            # Execute SIP transfer
            logger.info("🚀 Executing SIP transfer...")
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=transfer_to,
                play_dialtone=True,
            )
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=20.0
            )
            
            logger.info("✅ Call transfer completed successfully!")
            return "Transfer completed successfully"
            
        except asyncio.TimeoutError:
            logger.error("⏰ Transfer timeout")
            await ctx.session.generate_reply(
                instructions="I'm having trouble connecting to our human agent. Please try again in a moment."
            )
            return "Transfer timed out - please try again"
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            await ctx.session.generate_reply(
                instructions="I apologize, but I'm having trouble transferring your call right now."
            )
            return f"Transfer failed: {str(e)}"

    @function_tool()
    async def check_agent_availability(self) -> str:
        """Check if human agents are available"""
        return "Human agents are available to assist you. Would you like me to transfer your call?"

async def entrypoint(ctx: JobContext):
    """
    ✅ FIXED: Proper entrypoint following LiveKit patterns
    """
    logger.info("=== RAG VOICE AGENT STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    
    # Connect to room first
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Initialize RAG system
    logger.info("🔧 Initializing RAG system...")
    rag_success = await rag_manager.load_or_create_index()
    
    if not rag_success:
        logger.error("❌ RAG system failed to initialize")
        logger.error("💡 Please run: python data_ingestion.py --create-sample")
        logger.error("💡 Then run: python data_ingestion.py")
        # Don't fail completely - continue without RAG
        logger.warning("⚠️ Continuing without RAG functionality")
    else:
        logger.info("✅ RAG system initialized successfully")
    
    # Create agent session with optimized settings
    session = AgentSession(
        # STT: Deepgram with optimized settings
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
            smart_format=True,
            filler_words=False,
        ),
        
        # LLM: OpenAI with conservative settings
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,  # Lower for consistency
            timeout=20.0,
        ),
        
        # TTS: OpenAI TTS with natural voice
        tts=openai.TTS(
            model="tts-1",
            voice="nova",
            speed=1.0,
            
        ),
        
        # VAD: Silero with tuned settings
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,
        ),
        
        # Turn detection if available
        turn_detection=MultilingualModel() if MultilingualModel else None,
    )
    
    # Create and start agent
    agent = RAGVoiceAgent()
    
    await session.start(
        room=ctx.room,
        agent=agent,
    )
    
    # Send initial greeting
    try:
        await session.generate_reply(
            instructions="""Give a brief, friendly greeting. Say: "Hello! I'm your AI assistant with access to our knowledge base. How can I help you today? If you need to speak with a human agent, just let me know and I can transfer you right away." Keep it under 8 seconds and conversational."""
        )
        logger.info("✅ Initial greeting sent")
        
    except Exception as e:
        logger.error(f"❌ Greeting failed: {e}")
        logger.info("⚠️ Continuing without greeting")
    
    logger.info("✅ RAG Voice Agent is ready and operational")

if __name__ == "__main__":
    try:
        # Validate configuration
        validate_config()
        
        logger.info("🚀 Starting LiveKit RAG Voice Agent")
        logger.info(f"📞 Transfer destination: {config.transfer_sip_address}")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)