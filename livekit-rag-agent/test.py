import asyncio
from utils.streaming_rag_manager import streaming_rag_manager

async def test():
    success = await streaming_rag_manager.initialize()
    print(f'RAG system initialized: {success}')
    if success:
        result = await streaming_rag_manager.enhanced_query('What services do you offer?')
        print(f'Test query result: {result}')
