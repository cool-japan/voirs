-- VoiRS Recognizer Database Initialization
-- This script sets up the database schema for analytics and metadata

-- Create tables for recognition sessions
CREATE TABLE IF NOT EXISTS recognition_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_ms INTEGER,
    audio_length_ms INTEGER,
    model_used VARCHAR(100),
    features_used TEXT[],
    status VARCHAR(50) DEFAULT 'completed',
    error_message TEXT
);

-- Create table for performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES recognition_sessions(session_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create table for recognition results
CREATE TABLE IF NOT EXISTS recognition_results (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES recognition_sessions(session_id),
    text_result TEXT,
    confidence_score DECIMAL(5,4),
    language_detected VARCHAR(10),
    speaker_count INTEGER,
    segment_start_ms INTEGER,
    segment_end_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON recognition_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_model ON recognition_sessions(model_used);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON performance_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_results_session ON recognition_results(session_id);

-- Create a view for session summaries
CREATE OR REPLACE VIEW session_summaries AS
SELECT 
    rs.session_id,
    rs.created_at,
    rs.duration_ms,
    rs.audio_length_ms,
    rs.model_used,
    rs.status,
    COUNT(rr.id) as result_count,
    AVG(rr.confidence_score) as avg_confidence
FROM recognition_sessions rs
LEFT JOIN recognition_results rr ON rs.session_id = rr.session_id
GROUP BY rs.session_id, rs.created_at, rs.duration_ms, rs.audio_length_ms, rs.model_used, rs.status;

-- Insert sample data (for testing)
INSERT INTO recognition_sessions (session_id, duration_ms, audio_length_ms, model_used) 
VALUES 
    (gen_random_uuid(), 1500, 10000, 'whisper-base'),
    (gen_random_uuid(), 2300, 15000, 'whisper-small')
ON CONFLICT DO NOTHING;

COMMIT;
