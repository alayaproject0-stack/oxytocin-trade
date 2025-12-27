-- Supabase Schema for Oxytocin Trade
-- Run this in the Supabase SQL Editor to create the required tables

-- Portfolio state table
CREATE TABLE IF NOT EXISTS portfolio (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    balance DECIMAL(15, 2) NOT NULL DEFAULT 10000.00,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    shares DECIMAL(15, 6) NOT NULL,
    entry_price DECIMAL(15, 4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade history table
CREATE TABLE IF NOT EXISTS trade_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    action VARCHAR(20) NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    profit DECIMAL(15, 4) DEFAULT 0,
    confidence DECIMAL(5, 4),
    system2_used BOOLEAN DEFAULT FALSE,
    balance_after DECIMAL(15, 2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Summary view for dashboard
CREATE TABLE IF NOT EXISTS dashboard_summary (
    id INTEGER PRIMARY KEY DEFAULT 1,
    initial_balance DECIMAL(15, 2) DEFAULT 10000.00,
    current_balance DECIMAL(15, 2),
    total_value DECIMAL(15, 2),
    roi_pct DECIMAL(10, 4),
    last_update TIMESTAMPTZ DEFAULT NOW()
);

-- Insert initial portfolio record
INSERT INTO portfolio (balance) VALUES (10000.00)
ON CONFLICT DO NOTHING;

-- Insert initial summary record
INSERT INTO dashboard_summary (id, initial_balance, current_balance, total_value, roi_pct)
VALUES (1, 10000.00, 10000.00, 10000.00, 0.0)
ON CONFLICT (id) DO NOTHING;

-- Enable Row Level Security (optional, for added security)
ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE dashboard_summary ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations (adjust based on your needs)
CREATE POLICY "Allow all" ON portfolio FOR ALL USING (true);
CREATE POLICY "Allow all" ON positions FOR ALL USING (true);
CREATE POLICY "Allow all" ON trade_history FOR ALL USING (true);
CREATE POLICY "Allow all" ON dashboard_summary FOR ALL USING (true);
