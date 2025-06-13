"""
ULTRA-FAST: RAG Voice Agent with Sub-2-Second Response Times
FIXED: Better tool selection and transfer logic
"""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
import time

from livekit.plugins import elevenlabs


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

# ✅ SCALABLE RAG: Import the real system
from scalable_fast_rag import scalable_rag

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraFastAgent(Agent):
    """
    ✅ ULTRA-FAST: Agent optimized for sub-2-second responses
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI voice assistant for phone calls. 

CRITICAL INSTRUCTIONS:
- Keep responses very short (under 30 words) for phone clarity
- When you receive [FastRAG] information, use it to answer questions directly
- ONLY transfer to human when explicitly asked: "transfer me", "human agent", "speak to a person"
- For questions about "details", "more information", "tell me about" - use search_info or get_service_info
- NEVER transfer unless the user explicitly requests it

AVAILABLE TOOLS:
- get_service_info: Use for service-related questions
- search_info: Use for general information requests, details, explanations
- transfer_to_human: ONLY use when explicitly requested transfer

When user asks for "details" or "more information" - always use search_info, NOT transfer_to_human."""
        )
        self.processing = False
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        ⚡ ULTRA-FAST: RAG injection with <200ms target
        """
        try:
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 3 or self.processing:
                return
            
            self.processing = True
            
            try:
                # ✅ FIXED: More specific transfer detection
                explicit_transfer_phrases = [
                    "transfer me", 
                    "human agent", 
                    "speak to a person",
                    "talk to a human",
                    "connect me to someone",
                    "I want to speak to someone"
                ]
                
                # Skip RAG for EXPLICIT transfer requests only
                if any(phrase in user_text.lower() for phrase in explicit_transfer_phrases):
                    logger.info(f"🔄 Explicit transfer request detected: {user_text}")
                    return
                
                # ⚡ ULTRA-FAST: Get context with aggressive timeout
                results = await asyncio.wait_for(
                    scalable_rag.quick_search(user_text),
                    timeout=0.15  # 150ms max
                )
                
                if results and len(results) > 0:
                    # ✅ FIX: Clean content for voice response
                    raw_content = results[0]["content"]
                    context = self._clean_content_for_voice(raw_content)
                    turn_ctx.add_message(
                        role="system",
                        content=f"[FastRAG]: {context}"
                    )
                    logger.info("⚡ Ultra-fast RAG context injected")
                        
            except asyncio.TimeoutError:
                logger.debug("⚡ RAG timeout - continuing without context")
            except Exception as e:
                logger.error(f"❌ RAG error: {e}")
            finally:
                self.processing = False
                
        except Exception as e:
            logger.error(f"❌ on_user_turn_completed error: {e}")
            self.processing = False
    
    def _clean_content_for_voice(self, content: str) -> str:
        """Clean content for voice response"""
        try:
            # Remove Q: and A: prefixes for cleaner voice response
            content = content.replace("Q: ", "").replace("A: ", "")
            
            # Handle multi-line content - take first meaningful sentence
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines:
                # Take the first substantial line
                for line in lines:
                    if len(line) > 20 and not line.startswith(('Q:', 'A:', '#')):
                        content = line
                        break
                else:
                    content = lines[0]
            
            # Limit length for voice
            if len(content) > 200:
                content = content[:200].rsplit('.', 1)[0] + "."
            
            return content
            
        except Exception:
            return content[:150] if len(content) > 150 else content

    @function_tool()
    async def get_service_info(self, service_type: str = "general") -> str:
        """
        Get information about our services.
        
        Use this when users ask about:
        - What services do you offer
        - Service information
        - Business offerings
        """
        try:
            logger.info(f"🔍 Getting service info for: {service_type}")
            results = await scalable_rag.quick_search(f"services {service_type}")
            if results and len(results) > 0:
                content = self._clean_content_for_voice(results[0]["content"])
                return content
            else:
                return "We offer comprehensive AI voice assistant services including 24/7 customer support, automated information systems, and call routing. Would you like more specific details?"
        except Exception as e:
            logger.error(f"❌ Service info error: {e}")
            return "We provide AI voice assistant services. Would you like me to search for more specific information?"

    @function_tool()
    async def search_info(self, query: str) -> str:
        """
        Search for detailed information about any topic.
        
        Use this when users ask for:
        - More details
        - Additional information
        - Explanations
        - Specific questions about features, pricing, etc.
        
        DO NOT use transfer_to_human for information requests.
        """
        try:
            logger.info(f"🔍 Searching for detailed info: {query}")
            results = await scalable_rag.quick_search(query)
            if results and len(results) > 0:
                content = self._clean_content_for_voice(results[0]["content"])
                return content
            else:
                return "I can help with general information about our AI voice services, pricing, features, and integration options. What specific aspect would you like to know about?"
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return "I can provide information about our services. What specific details would you like to know?"

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """
        Transfer the caller to a human agent.
        
        ONLY use this when the caller EXPLICITLY requests:
        - "transfer me"
        - "human agent" 
        - "speak to a person"
        - "talk to a human"
        
        DO NOT use for information requests, details, or explanations.
        """
        try:
            logger.info("🔄 EXECUTING HUMAN TRANSFER - User explicitly requested")
            job_ctx = get_job_context()
            
            # Find SIP participant
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3":
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "I'm having trouble with the transfer. Please try calling back."
            
            # Quick transfer message
            speech_handle = ctx.session.generate_reply(
                instructions="Say: 'Connecting you to a human agent now. Please hold on.'"
            )
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=os.getenv("TRANSFER_SIP_ADDRESS", "sip:voiceai@sip.linphone.org"),
                play_dialtone=True,
            )
            
            await job_ctx.api.sip.transfer_sip_participant(transfer_request)
            return "Transfer to human agent completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "I'm having trouble with the transfer. Let me try to help you directly instead."

async def create_ultra_fast_session() -> AgentSession:
    """Create ultra-fast optimized session"""
    
    # ⚡ FAST TTS: Choose fastest option
    # if CARTESIA_AVAILABLE and os.getenv("CARTESIA_API_KEY"):
    #     tts_engine = cartesia.TTS(
    #         model="sonic-english",
    #         voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
    #         api_key=os.getenv("CARTESIA_API_KEY")
    #     )
    #     logger.info("🚀 Using Cartesia Sonic TTS (40ms)")
    # else:
    #     tts_engine = openai.TTS(
    #         model="tts-1",
    #         voice="nova",
    #         speed=1.1,  # Slightly faster
    #     )
    tts_engine=elevenlabs.TTS(
      voice_id="ODq5zmih8GrVes37Dizd",
      model="eleven_multilingual_v2"
   )
    logger.info("⚡ Using elevenlabs TTS")
    
    session = AgentSession(
        # ⚡ FAST STT: Use general model for reliability
        stt=deepgram.STT(
            model="nova-2-general",
            language="en",
        ),
        
        # ⚡ FAST LLM: Optimized settings
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,  # Slightly higher for more natural responses
        ),
        
        tts=tts_engine,
        
        # ⚡ FAST VAD: Default settings
        vad=silero.VAD.load(),
        
        turn_detection=MultilingualModel() if MultilingualModel else None,
    )
    
    return session

async def entrypoint(ctx: JobContext):
    """
    ⚡ ULTRA-FAST: Optimized entrypoint
    """
    logger.info("=== ULTRA-FAST RAG AGENT STARTING ===")
    
    # Connect immediately
    await ctx.connect()
    logger.info("✅ Connected")
    
    # ⚡ PARALLEL: Initialize everything at once
    init_tasks = [
        scalable_rag.initialize(),
        create_ultra_fast_session()
    ]
    
    rag_success, session = await asyncio.gather(*init_tasks)
    
    # Create agent
    agent = UltraFastAgent()
    
    # ⚡ START: Begin session immediately
    await session.start(room=ctx.room, agent=agent)
    
    # ✅ FIXED: Based on official LiveKit documentation examples
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )
    logger.info("✅ Initial greeting sent")
    
    logger.info("⚡ ULTRA-FAST AGENT READY!")
    logger.info(f"⚡ RAG Status: {'✅ Active' if rag_success else '⚠️ Fallback'}")

if __name__ == "__main__":
    try:
        logger.info("⚡ Starting Ultra-Fast RAG Agent")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)