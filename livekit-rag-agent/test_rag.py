import asyncio
from utils.rag_manager import rag_manager

async def test():
    success = await rag_manager.load_or_create_index()
    print(f'RAG system initialized: {success}')
    if success:
        # Test without timeout first
        try:
            nodes = await rag_manager.retriever.aretrieve('What services do you offer?')
            print(f'Retrieved {len(nodes)} nodes successfully!')
            for i, node in enumerate(nodes):
                content = node.get_content()[:100]
                print(f'Node {i}: {content}...')
        except Exception as e:
            print(f'Error: {e}')

asyncio.run(test())