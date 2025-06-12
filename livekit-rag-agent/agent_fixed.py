"""
Fixed LiveKit RAG Voice Agent
Uses proper LlamaIndex ChatEngine instead of manual RAG
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
    cli
)
from livekit.plugins import openai, deepgram, silero

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except ImportError:
    MultilingualModel = None

from dotenv import load_dotenv
load_dotenv()

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import config, validate_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProperRAGVoiceAgent(Agent):
    """
    ✅ FIXED: Proper RAG Voice Agent using LlamaIndex ChatEngine
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI voice assistant with access to a comprehensive knowledge base.

IMPORTANT GUIDELINES:
- Keep responses concise and conversational for voice (under 80 words)
- Use your knowledge base to provide specific, accurate information
- If someone asks to speak to a human, transfer them immediately
- Be friendly, professional, and helpful

TRANSFER KEYWORDS:
- "speak to a human" / "talk to a person"
- "transfer me" / "human agent"
- "customer service representative"
"""
        )
        self.chat_engine = None
        self.rag_ready = False
    
    async def initialize_rag_chat_engine(self):
        """Initialize proper LlamaIndex ChatEngine"""
        try:
            # Configure LlamaIndex
            Settings.embed_model = OpenAIEmbedding(
                model=config.embedding_model,
                api_key=config.openai_api_key
            )
            Settings.llm = OpenAI(
                model="gpt-4o-mini",
                api_key=config.openai_api_key,
                temperature=0.1,
                max_tokens=80,  # Short for voice
                timeout=10.0
            )
            
            # Load index
            storage_dir = config.storage_dir
            if storage_dir.exists() and any(storage_dir.iterdir()):
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                index = load_index_from_storage(storage_context)
                
                # ✅ PROPER: Use LlamaIndex ChatEngine
                memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
                self.chat_engine = index.as_chat_engine(
                    chat_mode="context",  # This handles RAG automatically
                    memory=memory,
                    system_prompt="""You are a helpful voice assistant with access to our knowledge base. 
                    
Keep responses very short for voice interaction (under 50 words). 
Use the context provided to give accurate, specific answers.
If asked about human agents or transfers, mention that transfers are available."""
                )
                
                self.rag_ready = True
                logger.info("✅ LlamaIndex ChatEngine initialized successfully")
                return True
            else:
                logger.error("❌ No RAG index found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChatEngine: {e}")
            return False
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        ✅ FIXED: Use LlamaIndex ChatEngine for proper RAG
        """
        try:
            if not self.chat_engine or not self.rag_ready:
                return
                
            user_text = new_message.text_content
            if not user_text or len(user_text.strip()) < 5:
                return
            
            # Check for transfer requests
            transfer_keywords = ["human", "person", "agent", "representative", "transfer"]
            if any(keyword in user_text.lower() for keyword in transfer_keywords):
                logger.info("🔄 Transfer request detected, skipping RAG")
                return
            
            # ✅ PROPER: Use LlamaIndex ChatEngine (handles RAG automatically)
            logger.info(f"🎯 Using ChatEngine for: {user_text[:50]}...")
            
            try:
                # Get response from ChatEngine (includes automatic RAG)
                response = await asyncio.wait_for(
                    self.chat_engine.achat(user_text),
                    timeout=3.0  # Fast timeout
                )
                
                # Inject the RAG-enhanced response into chat context
                rag_response = str(response).strip()
                if len(rag_response) > 150:
                    rag_response = rag_response[:150] + "..."
                
                turn_ctx.add_message(
                    role="assistant",
                    content=rag_response
                )
                logger.info("✅ Added ChatEngine response to conversation")
                
            except asyncio.TimeoutError:
                logger.warning("⏰ ChatEngine timeout")
            except Exception as e:
                logger.error(f"❌ ChatEngine error: {e}")
                
        except Exception as e:
            logger.error(f"❌ RAG processing error: {e}")
    
    @function_tool()
    async def search_knowledge_base(self, query: str) -> str:
        """Search knowledge base using ChatEngine"""
        try:
            if not self.chat_engine:
                return "Knowledge base not available."
                
            response = await asyncio.wait_for(
                self.chat_engine.achat(query),
                timeout=5.0
            )
            return str(response).strip()
            
        except Exception as e:
            logger.error(f"❌ Knowledge search error: {e}")
            return "Sorry, I encountered an error searching the knowledge base."
    
    @function_tool()
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer call to human agent"""
        try:
            job_ctx = get_job_context()
            transfer_to = config.transfer_sip_address
            
            # Find SIP participant
            sip_participant = None
            for participant in job_ctx.room.remote_participants.values():
                if str(participant.kind) == "3" or "sip_" in participant.identity.lower():
                    sip_participant = participant
                    break
            
            if not sip_participant:
                return "No active call found to transfer."
            
            # Inform user
            await ctx.session.generate_reply(
                instructions="Connecting you to a human agent now. Please hold."
            )
            
            # Execute transfer
            transfer_request = api.TransferSIPParticipantRequest(
                room_name=job_ctx.room.name,
                participant_identity=sip_participant.identity,
                transfer_to=transfer_to,
                play_dialtone=True,
            )
            
            await asyncio.wait_for(
                job_ctx.api.sip.transfer_sip_participant(transfer_request),
                timeout=15.0
            )
            
            return "Transfer completed successfully"
            
        except Exception as e:
            logger.error(f"❌ Transfer error: {e}")
            return "Transfer failed - please try again"

async def entrypoint(ctx: JobContext):
    """
    ✅ FIXED: Proper entrypoint with LlamaIndex integration
    """
    logger.info("=== PROPER RAG VOICE AGENT STARTING ===")
    logger.info(f"Room: {ctx.room.name}")
    
    await ctx.connect()
    logger.info("✅ Connected to room")
    
    # Create agent and initialize RAG
    agent = ProperRAGVoiceAgent()
    rag_success = await agent.initialize_rag_chat_engine()
    
    if not rag_success:
        logger.warning("⚠️ Continuing without RAG functionality")
    
    # Create optimized session
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.1,
            timeout=15.0,
        ),
        tts=openai.TTS(
            model="tts-1",
            voice="nova",
            speed=1,  # Faster speech
        ),
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.4,  # Faster response
        ),
        turn_detection=MultilingualModel() if MultilingualModel else None,
    )
    
    await session.start(room=ctx.room, agent=agent)
    
    # Quick greeting
    try:
        await session.generate_reply(
            instructions="Say: 'Hi! I'm your AI assistant. How can I help you?' Keep it under 3 seconds."
        )
        logger.info("✅ Quick greeting sent")
    except Exception as e:
        logger.error(f"❌ Greeting failed: {e}")
    
    logger.info("✅ Proper RAG Voice Agent ready!")

if __name__ == "__main__":
    try:
        validate_config()
        logger.info("🚀 Starting Proper RAG Voice Agent")
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="my-telephony-agent"
        ))
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        exit(1)