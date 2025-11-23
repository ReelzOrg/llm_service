from qdrant_client import QdrantClient, models

# Handle connections over gRPC
client = QdrantClient(url="http://localhost:6334")

# Create a collection
client.create_collection(
    collection_name="user_conversations",
    vectors_config=models.VectorParams(
        size=384, # we are using sentence-transformers/all-MiniLM-L6-v2
        distance="cosine"
    )
)

# Add a vector to the collection
# client.upsert(
#     collection_name="user_conversations",
#     points=[
#         models.PointStruct(
#             id="msg_123e4567-e89b-12d3-a456-426614174000",
#             vector=[0.123, -0.456, 0.789, ...],
#             payload={
#                 "user_id": "user_abc123",
#                 "conversation_id": "conv_xyz789",
#                 "message_type": "user",
#                 "content_preview": "How do I create a video with...",
#                 "timestamp": "2025-10-10T14:30:00Z",
#                 "session_id": "sess_def456",
#                 "intent": "question",
#                 "entities": ["video", "creation", "tutorial"],
#                 "language": "en"
#             }
#         )
#     ]
# )

# Search for similar vectors
# results = client.search(
#     collection_name="user_conversations",
#     query_vector=[0.123, -0.456, 0.789, ...],
#     limit=5
# )

print(results)
