import asyncio
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool
)
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ImageContent,
)
from livekit.plugins import deepgram, openai, silero


@function_tool
async def analyze_image(
    context: RunContext,
    user_msg: str,
):
    """Called when asked to evaluate something that would require vision capabilities,
    for example, an image, video, or the webcam feed."""
    print(f"Message triggering vision capabilities: {user_msg}")
    return {"analysis": "Vision analysis requested"}


class VisionAssistant(Agent):
    """Assistant with vision capabilities that can see and respond to video."""
    
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        
        super().__init__(
            instructions=(
                "Your name is Alloy. You are a funny, witty bot. Your interface with users will be voice and vision. "
                "Respond with short and concise answers. Avoid using unpronounceable punctuation or emojis. "
                "When you see an image in our conversation, naturally incorporate what you see into your response. "
                "Keep visual descriptions brief but informative."
            ),
            tools=[analyze_image],
        )

    async def on_enter(self):
        """Called when the agent enters the session."""
        room = agents.get_job_context().room
        
        # Find existing video tracks from remote participants
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if (publication.track and 
                    publication.track.kind == rtc.TrackKind.KIND_VIDEO):
                    self._create_video_stream(publication.track)
                    break
        
        # Watch for new video tracks
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Add the latest video frame to user messages when available."""
        if self._latest_frame:
            # Add the latest frame to the user's message
            if not hasattr(new_message, 'content') or new_message.content is None:
                new_message.content = []
            elif isinstance(new_message.content, str):
                new_message.content = [new_message.content]
            elif not isinstance(new_message.content, list):
                new_message.content = [new_message.content]
                
            new_message.content.append(ImageContent(image=self._latest_frame))
            print("Added latest video frame to user message")
            # Reset the frame so we don't reuse it
            self._latest_frame = None

    def _create_video_stream(self, track: rtc.Track):
        """Create a video stream to capture frames from the given track."""
        print(f"Creating video stream for track: {track.sid}")
        
        # Close any existing stream
        if self._video_stream is not None:
            self._video_stream.close()
        
        # Create new stream
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            """Continuously read frames and store the latest one."""
            try:
                async for event in self._video_stream:
                    # Store the latest frame for use later
                    self._latest_frame = event.frame
            except Exception as e:
                print(f"Error reading video stream: {e}")
        
        # Start the stream reading task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=rtc.room.AutoSubscribe.SUBSCRIBE_ALL)
    print(f"Room name: {ctx.room.name}")

    # Create session with STT, LLM, and TTS
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(voice="alloy"),
    )

    # Start the session with our vision-enabled assistant
    await session.start(agent=VisionAssistant(), room=ctx.room)
    
    # Initial greeting
    await session.say("Hi there! I can see and hear you. How can I help?")

    # Keep the session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))