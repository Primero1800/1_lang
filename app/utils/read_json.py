import asyncio
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from app.common.logging import log_decorator, logger

from qdrant_client.models import (
    PointStruct,
)

from app.core.config import settings
from app.dependencies.infrastructure import (
    get_vector_client,
    get_ai_client,
    get_aiohttp_session,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


@log_decorator(level=logging.INFO)
async def read_starting_json():

    time_started = time.time()
    logger.info(f"Script 'Read_starting_json' started: {time_started}")

    aiohttp_session = await get_aiohttp_session()
    ai_client = await get_ai_client(aiohttp_session)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", ",", " "],
    )

    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "govor.json"
    points_raw, points = [], []
    with open(data_path) as file:
        data = json.loads(file.read())
        messages = data.get("messages")

        logger.info(f"Found {len(messages)} messages...")
        logger.info("Processing....")

        for message in messages[:-2000]:
            entities = message.get("text_entities")
            if entities:
                full_text = []
                for entity in entities:
                    full_text.append(entity.get("text"))
                text_to_embed = "".join(full_text)
                if not text_to_embed:
                    continue

                chunks = splitter.split_text(text_to_embed)

                for idx, chunk in enumerate(chunks, start=1):
                    payload = {
                        "text": chunk,
                        "message_id": message.get("id"),
                        "chunk_id": idx,
                        "total_chunks": len(chunks),
                    }
                    points_raw.append(
                        {
                            "id": str(uuid4().hex),
                            "vector": chunk.lower(),
                            "payload": payload,
                        }
                    )

    to_embed = [item["vector"] for item in points_raw]
    logger.info(
        f"Making request for embedding creating... Wait please... {len(to_embed)} entities processing"
    )
    embeddings = await ai_client.embed(to_embed, task_type="document")
    if embeddings:
        logger.info(
            f"Embedding created successfully: {len(embeddings)} embeddings total"
        )
        logger.info("Adding new PointStruct for vector_client")
        for embedding, raw_point in zip(embeddings, points_raw):
            points.append(
                PointStruct(
                    id=raw_point["id"],
                    vector=embedding,
                    payload=raw_point["payload"],
                )
            )
        logger.info("PointStruct for vector_client added successfully")
    else:
        logger.error("Embedding creating failed")
        return False

    logger.info("Starting Vector database upserting....")
    vector_client = await get_vector_client()
    await vector_client.start()

    success = True
    try:
        if points:
            await vector_client.upsert(
                collection_name=settings.VECTOR_DB_COLLECTION,
                points=points,
                raise_exception=True,
            )
            logger.info("Vector database upserting successfully completed")
    except Exception:
        logger.warning("Vector database upserting error")
        success = False
    finally:
        exc_time = time.time() - time_started
        if success:
            logger.info(
                f"Script 'Read_starting_json' completed successfully in: {exc_time}s, {len(points)} vectors added"
            )
        else:
            logger.warning(f"Script 'Read_starting_json' failed in: {exc_time}s")


if __name__ == "__main__":
    asyncio.run(read_starting_json())
