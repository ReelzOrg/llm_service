-- Schema for LLM service persistence

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at DESC);