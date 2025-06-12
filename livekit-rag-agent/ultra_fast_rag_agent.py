"""
ULTRA-FAST: RAG Voice Agent with Sub-2-Second Response Times
Based on LiveKit official documentation and best practices
"""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
import time

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

CRITICAL: Keep responses very short (under 30 words) for phone clarity.

When you receive knowledge base information in [FastRAG], use it directly to answer questions.

For service questions, be specific about our 24/7 AI voice assistant services.

Only transfer to human when explicitly requested ("transfer me", "human agent")."""
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
                # Skip explicit transfer requests
                if any(phrase in user_text.lower() for phrase in ["transfer me", "human agent", "speak to person"]):
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
                        
            finally:
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
                
        except asyncio.TimeoutError:
            logger.debug("⚡ RAG timeout - continuing without context")
            self.processing = False
        except Exception as e:
            logger.error(f"❌ RAG error: {e}")
            self.processing = False

    @function_tool()
    async def get_service_info(self, service_type: str = "general") -> str:
        """Get service information quickly"""
        try:
            results = await scalable_rag.quick_search(f"services {service_type}")
            if results and len(results) > 0:
                content = self._clean_content_for_voice(results[0]["content"])
                return content
            else:
                return "We offer 24/7 AI voice assistant services. Would you like me to transfer you to a human agent for detailed information?"
        except Exception as e:
            logger.error(f"❌ Service info error: {e}")
            return "I can connect you with a human agent for service information."

    @function_tool()
    async def search_info(self, query: str) -> str:
        """Search for specific information"""
        try:
            results = await scalable_rag.quick_search(query)
            if results and len(results) > 0:
                content = self._clean_content_for_voice(results[0]["content"])
                return content
            else:
                return "I don't have specific information about that. Would you like me to transfer you to a human agent?"
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return "Let me connect you with a human agent who can help."

    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer to human agent"""
        try:
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
            await ctx.session.generate_reply(
                instructions="Say: 'Connecting you to an agent now.'"
            )
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=os.getenv("TRANSFER_SIP_ADDRESS", "sip:voiceai@sip.linphone.org"),
                play_dialtone=True,
            )
            
            await job_ctx.api.sip.transfer_sip_participant(transfer_request)
            return "Transfer completed"
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "Transfer failed - please try again"

async def create_ultra_fast_session() -> AgentSession:
    """Create ultra-fast optimized session"""
    
    # ⚡ FAST TTS: Choose fastest option
    if CARTESIA_AVAILABLE and os.getenv("CARTESIA_API_KEY"):
        tts_engine = cartesia.TTS(
            model="sonic-english",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
            api_key=os.getenv("CARTESIA_API_KEY")
        )
        logger.info("🚀 Using Cartesia Sonic TTS (40ms)")
    else:
        tts_engine = openai.TTS(
            model="tts-1",
            voice="nova",
            speed=1.1,  # Slightly faster
        )
        logger.info("⚡ Using OpenAI TTS")
    
    session = AgentSession(
        # ⚡ FAST STT: Use general model for reliability
        stt=deepgram.STT(
            model="nova-2-general",
            language="en",
        ),
        
        # ⚡ FAST LLM: Optimized settings
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.0,  # More deterministic = faster
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
        fast_rag.initialize(),
        create_ultra_fast_session()
    ]
    
    rag_success, session = await asyncio.gather(*init_tasks)
    
    # Create agent
    agent = UltraFastAgent()
    
    # ⚡ START: Begin session immediately
    await session.start(room=ctx.room, agent=agent)
    
    # ⚡ QUICK GREETING: Send immediately
    asyncio.create_task(
        session.generate_reply(instructions="Say: 'Hi! How can I help you today?'")
    )
    
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