-- ForgeStream ECEF Event Log
-- APPEND-ONLY: no updates, no deletes, ever.

CREATE TABLE IF NOT EXISTS events (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type       TEXT NOT NULL,
    parent_id        UUID REFERENCES events(id),
    branch_id        UUID NOT NULL,
    author           TEXT NOT NULL,
    evaluator        FLOAT NOT NULL,
    payload          JSONB NOT NULL,
    degradation_flag BOOLEAN DEFAULT FALSE,
    trust_region_ok  BOOLEAN DEFAULT TRUE
);

-- Append-only enforcement at database level
CREATE OR REPLACE RULE no_updates AS ON UPDATE TO events DO INSTEAD NOTHING;
CREATE OR REPLACE RULE no_deletes AS ON DELETE TO events DO INSTEAD NOTHING;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch_id);
CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_id);

-- GIN index for JSONB payload queries
CREATE INDEX IF NOT EXISTS idx_events_payload ON events USING GIN (payload);
