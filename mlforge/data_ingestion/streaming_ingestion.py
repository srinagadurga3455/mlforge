"""
streaming_ingestion.py
----------------------
Process streaming data that arrives in real time, chunk by chunk.

Use this for:
  - Live data from Kafka (transactions, events, sensor readings)
  - CSV files too large to load all at once

Each chunk is saved as a separate file as it arrives.

Usage:
    from mlforge.data_sources    import StreamLoader
    from mlforge.data_ingestion  import StreamingIngestion

    ingestion = StreamingIngestion(
        loader      = StreamLoader.from_kafka("transactions", ["localhost:9092"]),
        output_path = "data/streaming/",
        max_chunks  = 100,  # None = run forever
    )
    ingestion.run()
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StreamingIngestion:
    """Processes and saves streaming data chunk by chunk."""

    def __init__(self, loader, output_path: str,
                 max_chunks: int = None):
        self.loader      = loader
        self.output_path = output_path
        self.max_chunks  = max_chunks  # None = run forever
        os.makedirs(output_path, exist_ok=True)

    def run(self):
        """Process each chunk as it arrives and save to disk."""
        logger.info("Streaming ingestion started...")
        self.loader.connect()

        chunks_done = 0
        total_rows  = 0

        for chunk in self.loader.stream(max_chunks=self.max_chunks):
            # Save chunk as a separate file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"chunk_{chunks_done:04d}_{timestamp}.parquet"
            filepath  = os.path.join(self.output_path, filename)
            chunk.to_parquet(filepath, index=False)

            chunks_done += 1
            total_rows  += len(chunk)
            logger.info(f"Chunk {chunks_done}: {len(chunk):,} rows "
                        f"(total so far: {total_rows:,})")

        self.loader.close()
        logger.info(f"Stream complete — "
                    f"{chunks_done} chunks, {total_rows:,} total rows")
