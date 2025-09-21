-- VoiRS Feedback Database Initialization Script
-- This script sets up the initial database schema and configuration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS feedback;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO feedback, analytics, monitoring, public;

-- Users table
CREATE TABLE IF NOT EXISTS feedback.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    preferences JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+[.][A-Za-z]+$')
);

-- Sessions table
CREATE TABLE IF NOT EXISTS feedback.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES feedback.users(id) ON DELETE CASCADE,
    session_type VARCHAR(100) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    completed BOOLEAN DEFAULT false,
    completion_percentage DECIMAL(5,2) DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Feedback responses table
CREATE TABLE IF NOT EXISTS feedback.responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES feedback.sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES feedback.users(id) ON DELETE CASCADE,
    response_type VARCHAR(100) NOT NULL,
    content TEXT,
    score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- User progress tracking
CREATE TABLE IF NOT EXISTS feedback.user_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES feedback.users(id) ON DELETE CASCADE,
    skill_area VARCHAR(200) NOT NULL,
    current_level INTEGER DEFAULT 1,
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    achievements JSONB DEFAULT '[]'::jsonb,
    statistics JSONB DEFAULT '{}'::jsonb,
    UNIQUE(user_id, skill_area)
);

-- Analytics events
CREATE TABLE IF NOT EXISTS analytics.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES feedback.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES feedback.sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(200) NOT NULL,
    event_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Metrics storage
CREATE TABLE IF NOT EXISTS monitoring.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(200) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance logs
CREATE TABLE IF NOT EXISTS monitoring.performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_name VARCHAR(200) NOT NULL,
    duration_ms INTEGER NOT NULL,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON feedback.users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON feedback.users(username);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON feedback.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON feedback.sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_responses_session_id ON feedback.responses(session_id);
CREATE INDEX IF NOT EXISTS idx_responses_user_id ON feedback.responses(user_id);
CREATE INDEX IF NOT EXISTS idx_responses_created_at ON feedback.responses(created_at);
CREATE INDEX IF NOT EXISTS idx_progress_user_id ON feedback.user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_events_user_id ON analytics.events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON analytics.events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON analytics.events(event_type);
CREATE INDEX IF NOT EXISTS idx_metrics_name_recorded ON monitoring.metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_performance_operation ON monitoring.performance_logs(operation_name);
CREATE INDEX IF NOT EXISTS idx_performance_created_at ON monitoring.performance_logs(created_at);

-- GIN indexes for JSONB fields
CREATE INDEX IF NOT EXISTS idx_users_preferences ON feedback.users USING GIN(preferences);
CREATE INDEX IF NOT EXISTS idx_sessions_metadata ON feedback.sessions USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_responses_metadata ON feedback.responses USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_progress_achievements ON feedback.user_progress USING GIN(achievements);
CREATE INDEX IF NOT EXISTS idx_progress_statistics ON feedback.user_progress USING GIN(statistics);
CREATE INDEX IF NOT EXISTS idx_events_data ON analytics.events USING GIN(event_data);
CREATE INDEX IF NOT EXISTS idx_metrics_labels ON monitoring.metrics USING GIN(labels);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON feedback.users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_progress_updated_at 
    BEFORE UPDATE ON feedback.user_progress 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for development
INSERT INTO feedback.users (username, email, preferences) VALUES
    ('test_user', 'test@example.com', '{"theme": "dark", "notifications": true}'),
    ('demo_user', 'demo@example.com', '{"theme": "light", "notifications": false}')
ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA feedback TO voirs;
GRANT USAGE ON SCHEMA analytics TO voirs;
GRANT USAGE ON SCHEMA monitoring TO voirs;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA feedback TO voirs;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO voirs;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO voirs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA feedback TO voirs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO voirs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO voirs;