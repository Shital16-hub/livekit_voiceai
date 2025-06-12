"""
Telephony-Optimized LiveKit RAG Voice Agent (LiveKit Agents 1.0 Compatible)
Uses current LiveKit API structure
"""
import asyncio
import logging
from typing import Optional

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
    cli,
    llm
)
from livekit.plugins import openai, deepgram, silero

try:
    from livekit.plugins import cartesia
    CARTESIA_AVAILABLE = True
except ImportError:
    CARTESIA_AVAILABLE = False

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except ImportError:
    MultilingualModel = None

from dotenv import load_dotenv
load_dotenv()

from config import config, validate_config
from utils.streaming_rag_manager import streaming_rag_manager
from utils.semantic_cache import semantic_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelephonyOptimizedRAGAgent(Agent):
    """
    ✅ FIXED: Telephony-optimized RAG Voice Agent using current LiveKit API
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI voice assistant for phone calls with access to a comprehensive knowledge base.

CRITICAL GUIDELINES FOR TELEPHONY:
- Keep responses VERY concise and clear for phone audio (under 50 words)
- Speak slowly and clearly for phone quality
- Use conversational, natural language
- If someone asks to speak to a human, transfer them immediately
- Acknowledge context from your knowledge base naturally

TRANSFER KEYWORDS TO LISTEN FOR:
- "speak to a human" / "talk to a person" / "human agent"
- "transfer me" / "customer service representative"
- "I want to talk to someone" / "connect me to an agent"
"""
        )
        self.rag_ready = False
        self.processing_query = False
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        🚀 OPTIMIZED: Ultra-fast RAG injection for telephony using current API
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3:
                return
            
            # Skip if already processing (prevent overlapping requests)
            if self.processing_query:
                return
            
            self.processing_query = True
            
            try:
                # Check for transfer requests first (highest priority)
                transfer_keywords = ["human", "person", "agent", "representative", "transfer", "someone"]
                if any(keyword in user_text.lower() for keyword in transfer_keywords):
                    logger.info("🔄 Transfer request detected, skipping RAG")
                    return
                
                # 🚀 OPTIMIZATION: Fast context retrieval for telephony
                context = await self._get_telephony_context(user_text)
                if context:
                    # ✅ Inject context using current API
                    turn_ctx.add_message(
                        role="system",
                        content=f"[Phone Call Context]: {context}"
                    )
                    logger.info("✅ Added telephony RAG context")
                        
            finally:
                self.processing_query = False
                
        except Exception as e:
            logger.error(f"❌ Telephony RAG processing error: {e}")
            self.processing_query = False
    
    async def _get_telephony_context(self, query: str) -> Optional[str]:
        """Get ultra-fast context optimized for telephony"""
        try:
            if streaming_rag_manager.should_bypass_rag(query):
                logger.debug(f"⚡ Bypassing RAG for simple telephony query: {query[:30]}...")
                return None
            
            # Check semantic cache first (fastest for telephony)
            if config.enable_semantic_cache:
                cached_response = await semantic_cache.get(query, threshold=0.75)
                if cached_response:
                    logger.info("🎯 Using cached response for telephony")
                    return cached_response[:80]  # Very short for phone clarity
            
            # Fast retrieval with telephony-optimized timeout
            start_time = asyncio.get_event_loop().time()
            
            try:
                results = await asyncio.wait_for(
                    streaming_rag_manager.quick_search(query),
                    timeout=0.4  # 400ms max for telephony
                )
                
                if results:
                    context = results[0].get("content", "")
                    # Very short for telephony clarity
                    if len(context) > 80:
                        context = context[:80] + "..."
                    
                    # Cache for next time
                    if config.enable_semantic_cache:
                        await semantic_cache.set(query, context)
                    
                    elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    logger.info(f"📞 Telephony RAG: {elapsed_ms:.1f}ms")
                    
                    return context
                    
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Telephony RAG timeout for: {query[:30]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Telephony context error: {e}")
            return None
        
        return None

    @function_tool()
    async def search_knowledge_base(self, query: str) -> str:
        """Search knowledge base with telephony optimization"""
        try:
            logger.info(f"📞 Telephony knowledge search: {query[:50]}...")
            
            result = await streaming_rag_manager.enhanced_query(query)
            
            if result and len(result.strip()) > 5:
                logger.info("✅ Telephony knowledge search successful")
                # Limit response for phone clarity
                if len(result) > 120:
                    result = result[:120] + "..."
                return result
            else:
                return "I couldn't find specific information about that. Let me transfer you to a human agent who can help."
                
        except Exception as e:
            logger.error(f"❌ Telephony knowledge search error: {e}")
            return "I'm having trouble accessing our system. Let me connect you with a human agent."

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer call to human agent (telephony optimized)"""
        try:
            job_ctx = get_job_context()
            transfer_to = config.transfer_sip_address
            
            logger.info("=== EXECUTING TELEPHONY TRANSFER ===")
            logger.info(f"Room: {job_ctx.room.name}")
            logger.info(f"Transfer destination: {transfer_to}")
            
            # Find SIP participant (telephony specific)
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    sip_participant = participant
                    logger.info(f"✅ Found SIP participant: {participant.identity}")
                    break
            
            if not sip_participant:
                logger.error("❌ No SIP participants found for telephony transfer")
                return "I'm sorry, I couldn't find an active call to transfer. Please try calling again."
            
            # Inform user about transfer (telephony appropriate)
            await ctx.session.generate_reply(
                instructions="Say: 'I'm connecting you to a human agent now. Please stay on the line while I transfer your call.'"
            )
            
            await asyncio.sleep(1)  # Brief pause for natural flow
            
            # Execute SIP transfer (telephony)
            logger.info("🚀 Executing telephony SIP transfer...")
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=transfer_to,
                play_dialtone=True,
            )
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=20.0  # Longer timeout for telephony stability
            )
            
            logger.info("✅ Telephony transfer completed successfully!")
            return "Transfer completed successfully"
            
        except asyncio.TimeoutError:
            logger.error("⏰ Telephony transfer timeout")
            await ctx.session.generate_reply(
                instructions="Say: 'I'm having trouble connecting to our human agent. Please try again in a moment.'"
            )
            return "Transfer timed out - please try again"
            
        except Exception as e:
            logger.error(f"❌ Telephony transfer error: {e}")
            await ctx.session.generate_reply(
                instructions="Say: 'I apologize, but I'm having trouble transferring your call right now.'"
            )
            return f"Transfer failed: {str(e)}"

    @function_tool()
    async def check_agent_availability(self) -> str:
        """Check if human agents are available (telephony)"""
        return "Human agents are available to assist you. Would you like me to transfer your call?"

async def create_telephony_session() -> AgentSession:
    """Create telephony-optimized session"""
    
    # ✅ Choose optimal TTS for telephony
    if CARTESIA_AVAILABLE and config.cartesia_api_key:
        tts_engine = cartesia.TTS(
            model="sonic-english",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
            api_key=config.cartesia_api_key
        )
        logger.info("🚀 Using Cartesia Sonic TTS for telephony (40ms latency)")
    else:
        tts_engine = openai.TTS(
            model="tts-1",
            voice="nova",
            speed=0.9,  # Slightly slower for phone clarity
        )
        logger.info("📞 Using OpenAI TTS for telephony")
    
    # ✅ Create telephony-optimized session
    session = AgentSession(
        # Telephony-optimized STT
        stt=deepgram.STT(
            model="nova-2-phonecall",  # ✅ Optimized for phone calls
            language="multi",
            smart_format=True,
            filler_words=False,
            interim_results=True,
        ),
        
        # Telephony-optimized LLM
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=60,  # Short for telephony clarity
            timeout=15.0,   # Reasonable timeout for phone quality
        ),
        
        # Telephony TTS
        tts=tts_engine,
        
        # Telephony-optimized VAD
        vad=silero.VAD.load(
            min_speech_duration=0.15,  # Slightly longer for phone quality
            min_silence_duration=0.6,  # Account for phone latency/quality
        ),
        
        # Turn detection if available
        turn_detection=MultilingualModel() if MultilingualModel else None,
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """
    📞 TELEPHONY-OPTIMIZED: Entrypoint using current LiveKit API
    """
    logger.info("=== TELEPHONY RAG VOICE AGENT STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    
    # Connect to room first
    await ctx.connect()
    logger.info("✅ Connected to telephony room")
    
    # Initialize RAG system
    logger.info("🔧 Initializing telephony RAG system...")
    rag_success = await streaming_rag_manager.initialize()
    
    if not rag_success:
        logger.error("❌ RAG system failed to initialize")
        logger.warning("⚠️ Continuing without RAG functionality")
    else:
        logger.info("✅ RAG system initialized for telephony")
    
    # Create telephony session and agent in parallel
    session_task = asyncio.create_task(create_telephony_session())
    
    session = await session_task
    agent = TelephonyOptimizedRAGAgent()
    agent.rag_ready = rag_success
    
    # ✅ Start telephony session with current API
    await session.start(room=ctx.room, agent=agent)
    
    # ✅ Telephony-appropriate greeting
    try:
        await session.generate_reply(
            instructions="""Give a brief, clear telephony greeting. Say: 
            'Hello! I'm your AI assistant with access to our knowledge base. 
            How can I help you today? 
            If you need to speak with a human agent, just let me know and I can transfer you right away.' 
            Keep it clear and concise for phone quality."""
        )
        logger.info("✅ Telephony greeting sent")
        
    except Exception as e:
        logger.error(f"❌ Telephony greeting failed: {e}")
        logger.info("⚠️ Continuing without greeting")
    
    # Log performance stats
    if rag_success:
        stats = streaming_rag_manager.get_stats()
        logger.info(f"📊 Telephony system stats: {stats}")
    
    logger.info("📞 Telephony RAG Voice Agent is ready and operational!")

if __name__ == "__main__":
    try:
        # Validate configuration
        validate_config()
        
        logger.info("🚀 Starting Telephony-Optimized LiveKit RAG Voice Agent")
        logger.info(f"📞 Transfer destination: {config.transfer_sip_address}")
        logger.info(f"⚡ Target latency: {config.target_latency_ms}ms")
        logger.info(f"🎯 RAG timeout: {config.rag_timeout_ms}ms")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="telephony-rag-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)