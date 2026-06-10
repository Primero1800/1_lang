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


@log_decorator(level=logging.INFO)
async def read_starting_json():

    time_started = time.time()
    logger.info(f"Script 'Read_starting_json' started: {time_started}")

    aiohttp_session = await get_aiohttp_session()
    ai_client = await get_ai_client(aiohttp_session)

    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "govor.json"
    points = []
    with open(data_path) as file:
        data = json.loads(file.read())
        messages = data.get("messages")

        logger.info(f"Found {len(messages)} messages...")
        logger.info("Processing....")

        for message in messages[-10:]:
            entities = message.get("text_entities")
            if entities:
                full_text = []
                for entity in entities:
                    if entity.get("type") == "plain":
                        full_text.append(entity.get("text"))
                text_to_embed = " ".join(full_text)

                payload = {"context": text_to_embed, "metadata": message.get("id")}

                logger.info("Making request for embedding creating... Wait please...")

                embeddings = await ai_client.embed(text_to_embed)

                if embeddings:
                    logger.info("Embedding created successfully")
                    logger.info("Adding new PointStruct for vector_client")

                    points.append(
                        PointStruct(
                            id=str(uuid4().hex),
                            vector=embeddings,
                            payload=payload,
                        )
                    )
                    logger.info("PointStruct for vector_client added successfully")
                else:
                    logger.error("Embedding creating failed")

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
