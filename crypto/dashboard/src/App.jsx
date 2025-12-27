import React, { useMemo, useState, useEffect } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, Legend, LineChart, Line
} from 'recharts';
import { TrendingUp, Activity, Cpu, ShieldCheck, Zap, BarChart3, Clock, Download, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';
import './index.css';
import { supabase } from './supabaseClient';

// CSV Export function
const exportToCSV = (data, filename = 'trade_history.csv') => {
    const headers = ['Date', 'Ticker', 'Action', 'Price', 'Profit', 'Confidence', 'System2_Used', 'Balance'];
    const rows = data.daily_data.map(trade => [
        trade.date || trade.created_at?.split('T')[0],
        trade.ticker || 'TDK',
        trade.action,
        (trade.price || 0).toFixed(2),
        (trade.profit || 0).toFixed(2),
        (trade.confidence || 0.5).toFixed(4),
        trade.system2_used ? 'Yes' : 'No',
        (trade.balance_after || trade.balance || 0).toFixed(2)
    ]);

    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};

const StatCard = ({ title, value, icon: Icon, color }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card"
    >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="stat-label">{title}</div>
            <div style={{ padding: '10px', borderRadius: '14px', background: `${color}15`, color: color }}>
                <Icon size={20} />
            </div>
        </div>
        <div className="stat-value">{value}</div>
    </motion.div>
);

const INITIAL_BALANCE = 1000000;

const App = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState(null);

    const fetchData = async () => {
        setLoading(true);
        try {
            if (supabase) {
                const [tradesRes, summaryRes] = await Promise.all([
                    supabase.from('trade_history').select('*').order('created_at', { ascending: false }).limit(5000),
                    supabase.from('dashboard_summary').select('*').eq('id', 1).single()
                ]);
                if (tradesRes.error) throw tradesRes.error;
                const trades = tradesRes.data || [];
                const summary = summaryRes.data || {};
                const totalTrades = trades.filter(t => t.action === 'BUY' || t.action === 'SELL').length;
                const successfulTrades = trades.filter(t => t.profit > 0).length;
                const system2Trades = trades.filter(t => t.system2_used).length;
                setData({
                    summary: {
                        ticker: 'Multi', period: 'Live',
                        initial_balance: summary.initial_balance || INITIAL_BALANCE,
                        final_balance: summary.current_balance || (trades.length > 0 ? trades[trades.length - 1].balance_after : INITIAL_BALANCE),
                        current_balance: summary.current_balance || INITIAL_BALANCE,
                        total_value: summary.total_value || INITIAL_BALANCE,
                        roi_pct: summary.roi_pct || 0,
                        accuracy_pct: totalTrades > 0 ? (successfulTrades / totalTrades * 100) : 0,
                        system2_wake_rate_pct: trades.length > 0 ? (system2Trades / trades.length * 100) : 0,
                        energy_saved_pct: trades.length > 0 ? ((trades.length - system2Trades) / trades.length * 100) : 100
                    },
                    daily_data: trades.map(t => ({
                        date: t.created_at?.split('T')[0], ticker: t.ticker, action: t.action, price: t.price,
                        profit: t.profit, confidence: t.confidence || 0.5, system2_used: t.system2_used,
                        correct: t.profit >= 0, balance: t.balance_after
                    }))
                });
                setLastUpdate(new Date().toLocaleTimeString('ja-JP'));
            } else {
                const res = await fetch('/data.json');
                const json = await res.json();
                setData(json);
            }
        } catch (err) {
            console.error("Error loading data:", err);
            try { const res = await fetch('/data.json'); setData(await res.json()); } catch (e) { console.error(e); }
        }
        setLoading(false);
    };

    useEffect(() => {
        fetchData();
        if (supabase) {
            const interval = setInterval(fetchData, 60000);
            return () => clearInterval(interval);
        }
    }, []);

    const chartData = useMemo(() => {
        if (!data) return [];
        return data.daily_data.map(d => ({
            ...d,
            balance: Math.round(d.balance),
            confidence: Math.round(d.confidence * 100)
        }));
    }, [data]);

    const [sortConfig, setSortConfig] = useState({ key: 'date', direction: 'desc' });

    const requestSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const getSortIndicator = (name) => {
        if (sortConfig.key === name) {
            return sortConfig.direction === 'asc' ? ' ↑' : ' ↓';
        }
        return '';
    };

    const sortedData = useMemo(() => {
        if (!data) return [];
        let sortableData = [...data.daily_data];
        if (sortConfig.key !== null) {
            sortableData.sort((a, b) => {
                let aValue = a[sortConfig.key];
                let bValue = b[sortConfig.key];

                // Handle special cases
                if (sortConfig.key === 'profit') aValue = parseFloat(aValue || 0);
                if (sortConfig.key === 'price') aValue = parseFloat(aValue || 0);

                if (aValue < bValue) {
                    return sortConfig.direction === 'asc' ? -1 : 1;
                }
                if (aValue > bValue) {
                    return sortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableData;
    }, [data, sortConfig]);

    if (loading) return <div style={{ color: '#fff', textAlign: 'center', padding: '100px' }}>Loading Live Data...</div>;
    if (!data) return <div style={{ color: '#fff', textAlign: 'center', padding: '100px' }}>No Data Found. Please run simulation.</div>;

    const { summary } = data;

    // Configured Initial Balance
    const displayInitial = INITIAL_BALANCE; // 1,000,000

    // Check if DB data is from the old '10k' era
    // If DB's initial balance is significantly different (e.g., < 100,000), assume it's old data
    const isOldData = (summary.initial_balance || 0) < 500000;

    // If old data, visualize a "Fresh Start" (1M flat) until new data comes in
    // Otherwise use the livedata
    const displayCurrent = isOldData ? INITIAL_BALANCE : (summary.final_balance || summary.current_balance || 0);
    const displayProfit = displayCurrent - displayInitial;
    const displayRoi = (displayProfit / displayInitial) * 100;

    const pieData = [
        { name: 'System 1 (SNN)', value: parseFloat((100 - summary.system2_wake_rate_pct).toFixed(1)) },
        { name: 'System 2 (FinBERT)', value: parseFloat(summary.system2_wake_rate_pct.toFixed(1)) },
    ];

    const COLORS = ['#00d2ff', '#ff00c1'];

    return (
        <div className="dashboard-container">
            <header className="header">
                <div className="title-section">
                    <h1>Oxytocin Trade</h1>
                    <div style={{ color: '#718096', marginTop: '4px', fontWeight: 500 }}>
                        Hybrid Intelligence Performance Tracker
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    {lastUpdate && (
                        <div style={{ color: '#718096', fontSize: '0.85rem' }}>
                            更新: {lastUpdate}
                        </div>
                    )}
                    <button
                        onClick={fetchData}
                        disabled={loading}
                        style={{
                            display: 'flex', alignItems: 'center', gap: '0.5rem',
                            padding: '0.5rem 1rem', background: 'rgba(0,210,255,0.1)',
                            border: '1px solid rgba(0,210,255,0.3)', borderRadius: '8px',
                            color: '#00d2ff', cursor: 'pointer', fontSize: '0.85rem'
                        }}
                    >
                        <RefreshCw size={16} className={loading ? 'spin' : ''} />
                        更新
                    </button>
                    <div className="glass-card" style={{ padding: '0.6rem 1.25rem', display: 'flex', gap: '1rem', alignItems: 'center', borderRadius: '16px' }}>
                        <Clock size={18} color="#00d2ff" />
                        <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{summary.ticker} • {summary.period}</span>
                    </div>
                </div>
            </header>

            {/* Portfolio Summary - 元本と損益 */}
            <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card"
                style={{ marginBottom: '1.5rem', padding: '1.5rem' }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div className="stat-label" style={{ fontSize: '1.1rem' }}>Portfolio Summary</div>
                    <TrendingUp size={20} color={displayRoi >= 0 ? '#00f5d4' : '#ff4d4d'} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem' }}>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>元本 (Initial)</div>
                        <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff' }}>
                            ¥{displayInitial.toLocaleString()}
                        </div>
                    </div>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>現在価値 (Current)</div>
                        <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#00d2ff' }}>
                            {/* If outdated, show Initial, else show DB value */}
                            ¥{displayCurrent.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                        </div>
                    </div>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>損益 (Profit/Loss)</div>
                        <div style={{
                            fontSize: '1.8rem',
                            fontWeight: 700,
                            color: displayProfit >= 0 ? '#00f5d4' : '#ff4d4d'
                        }}>
                            {displayProfit >= 0 ? '+' : ''}
                            ¥{displayProfit.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                        </div>
                    </div>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>ROI</div>
                        <div style={{
                            fontSize: '1.8rem',
                            fontWeight: 700,
                            color: displayRoi >= 0 ? '#00f5d4' : '#ff4d4d'
                        }}>
                            {displayRoi >= 0 ? '+' : ''}{displayRoi.toFixed(2)}%
                        </div>
                    </div>
                </div>
            </motion.section>

            <section className="stats-grid">
                <StatCard title="Accuracy" value={`${summary.accuracy_pct.toFixed(1)}%`} icon={ShieldCheck} color="#3aedff" />
                <StatCard title="S2 Usage" value={`${summary.system2_wake_rate_pct.toFixed(1)}%`} icon={Zap} color="#ff00c1" />
                <StatCard title="Efficiency" value={`${summary.energy_saved_pct.toFixed(1)}%`} icon={Cpu} color="#00d2ff" />
                <StatCard title="Total Trades" value={data.daily_data.length} icon={Activity} color="#00f5d4" />
            </section>

            <div className="charts-grid">
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass-card chart-container">
                    <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div className="stat-label">Equity Curve (JPY)</div>
                        <BarChart3 size={18} color="#00d2ff" />
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={chartData}>
                            <defs>
                                <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#00d2ff" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#00d2ff" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="date" hide />
                            <YAxis domain={['auto', 'auto']} stroke="rgba(255,255,255,0.3)" />
                            <Tooltip
                                contentStyle={{ background: '#0a0e14', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.5)' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Area type="monotone" dataKey="balance" stroke="#00d2ff" fillOpacity={1} fill="url(#colorBalance)" strokeWidth={3} />
                        </AreaChart>
                    </ResponsiveContainer>
                </motion.div>

                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }} className="glass-card">
                    <div className="stat-label" style={{ marginBottom: '2rem' }}>Inference Mix</div>
                    <ResponsiveContainer width="100%" height={260}>
                        <PieChart>
                            <Pie data={pieData} innerRadius={65} outerRadius={85} paddingAngle={8} dataKey="value">
                                {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                            </Pie>
                            <Tooltip contentStyle={{ background: '#0a0e14', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }} />
                            <Legend verticalAlign="bottom" align="center" iconType="circle" />
                        </PieChart>
                    </ResponsiveContainer>
                </motion.div>
            </div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card">
                <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div className="stat-label">Model Confidence Timeline</div>
                    <Activity size={18} color="#ff00c1" />
                </div>
                <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={chartData}>
                        <XAxis dataKey="date" hide />
                        <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.3)" />
                        <Tooltip contentStyle={{ background: '#0a0e14', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }} />
                        <Line type="stepAfter" dataKey="confidence" stroke="#ff00c1" strokeWidth={2} dot={false} strokeDasharray="5 5" />
                    </LineChart>
                </ResponsiveContainer>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div className="stat-label">全取引ログ (All Trades)</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <span style={{ color: '#718096', fontSize: '0.85rem' }}>{data.daily_data.length} 件</span>
                        <button
                            onClick={() => exportToCSV(data)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                padding: '0.5rem 1rem',
                                background: 'linear-gradient(135deg, #00d2ff 0%, #3aedff 100%)',
                                border: 'none',
                                borderRadius: '8px',
                                color: '#0a0e14',
                                fontWeight: 600,
                                cursor: 'pointer',
                                fontSize: '0.85rem'
                            }}
                        >
                            <Download size={16} />
                            CSV出力
                        </button>
                    </div>
                </div>
                <div className="trade-table-container" style={{ maxHeight: '500px', overflowY: 'auto' }}>
                    <table className="trade-table" style={{ position: 'relative' }}>
                        <thead style={{ position: 'sticky', top: 0, background: '#0f1419', zIndex: 1 }}>
                            <tr>
                                <th onClick={() => requestSort('date')} style={{ cursor: 'pointer' }}>Date{getSortIndicator('date')}</th>
                                <th onClick={() => requestSort('ticker')} style={{ cursor: 'pointer' }}>Ticker{getSortIndicator('ticker')}</th>
                                <th onClick={() => requestSort('action')} style={{ cursor: 'pointer' }}>Action{getSortIndicator('action')}</th>
                                <th onClick={() => requestSort('price')} style={{ cursor: 'pointer' }}>Price{getSortIndicator('price')}</th>
                                <th>Shares</th>
                                <th onClick={() => requestSort('profit')} style={{ cursor: 'pointer' }}>Profit{getSortIndicator('profit')}</th>
                                <th>System</th>
                                <th>Success</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedData.map((trade, i) => {
                                // Calculate estimated shares (using balance * 0.6 / price as used in live_trader)
                                const estimatedShares = trade.shares || (trade.action === 'BUY' ? ((trade.balance || 1000000) * 0.6 / (trade.price || 1)).toFixed(4) : '-');
                                return (
                                    <tr key={i}>
                                        <td>{trade.date}</td>
                                        <td style={{ color: '#00d2ff', fontWeight: 600 }}>{trade.ticker || '-'}</td>
                                        <td>
                                            <span className={`badge badge-${(trade.action || 'hold').toLowerCase()}`}>
                                                {trade.action}
                                            </span>
                                        </td>
                                        <td>¥{(trade.price || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                                        <td style={{ color: '#718096' }}>
                                            {trade.action === 'BUY' || trade.action === 'SELL' ? estimatedShares : '-'}
                                        </td>
                                        <td style={{ color: (trade.profit || 0) >= 0 ? '#00f5d4' : '#ff4d4d', fontWeight: 600 }}>
                                            {(trade.profit || 0) >= 0 ? '+' : ''}¥{Math.round(trade.profit || 0).toLocaleString()}
                                        </td>
                                        <td>
                                            {trade.system2_used ?
                                                <span className="badge badge-s2">System 2</span> :
                                                <span className="badge" style={{ background: 'rgba(0,210,255,0.1)', color: '#00d2ff' }}>System 1</span>
                                            }
                                        </td>
                                        <td style={{ color: trade.correct ? '#00f5d4' : '#ff4d4d' }}>
                                            {trade.correct ? '✓ Success' : '✗ Miss'}
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </motion.div>
        </div>
    );
};

export default App;
