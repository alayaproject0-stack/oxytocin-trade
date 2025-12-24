import React, { useMemo, useState, useEffect } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, Legend, LineChart, Line
} from 'recharts';
import { TrendingUp, Activity, Cpu, ShieldCheck, Zap, BarChart3, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import './index.css';

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

const App = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/data.json')
            .then(res => res.json())
            .then(json => {
                setData(json);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error loading data:", err);
                setLoading(false);
            });
    }, []);

    const chartData = useMemo(() => {
        if (!data) return [];
        return data.daily_data.map(d => ({
            ...d,
            balance: Math.round(d.balance),
            confidence: Math.round(d.confidence * 100)
        }));
    }, [data]);

    if (loading) return <div style={{ color: '#fff', textAlign: 'center', padding: '100px' }}>Loading Live Data...</div>;
    if (!data) return <div style={{ color: '#fff', textAlign: 'center', padding: '100px' }}>No Data Found. Please run simulation.</div>;

    const { summary } = data;

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
                <div className="glass-card" style={{ padding: '0.6rem 1.25rem', display: 'flex', gap: '1rem', alignItems: 'center', borderRadius: '16px' }}>
                    <Clock size={18} color="#00d2ff" />
                    <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{summary.ticker} • {summary.period}</span>
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
                    <TrendingUp size={20} color={summary.roi_pct >= 0 ? '#00f5d4' : '#ff4d4d'} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem' }}>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>元本 (Initial)</div>
                        <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#fff' }}>
                            ¥{(summary.initial_balance || 10000).toLocaleString()}
                        </div>
                    </div>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>現在価値 (Current)</div>
                        <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#00d2ff' }}>
                            ¥{(summary.final_balance || summary.current_balance || 10000).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                        </div>
                    </div>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>損益 (Profit/Loss)</div>
                        <div style={{
                            fontSize: '1.8rem',
                            fontWeight: 700,
                            color: ((summary.final_balance || summary.current_balance || 10000) - (summary.initial_balance || 10000)) >= 0 ? '#00f5d4' : '#ff4d4d'
                        }}>
                            {((summary.final_balance || summary.current_balance || 10000) - (summary.initial_balance || 10000)) >= 0 ? '+' : ''}
                            ¥{((summary.final_balance || summary.current_balance || 10000) - (summary.initial_balance || 10000)).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                        </div>
                    </div>
                    <div>
                        <div style={{ color: '#718096', fontSize: '0.85rem', marginBottom: '0.5rem' }}>ROI</div>
                        <div style={{
                            fontSize: '1.8rem',
                            fontWeight: 700,
                            color: summary.roi_pct >= 0 ? '#00f5d4' : '#ff4d4d'
                        }}>
                            {summary.roi_pct >= 0 ? '+' : ''}{summary.roi_pct.toFixed(2)}%
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
                <div className="stat-label">Recent Trading Activity</div>
                <div className="trade-table-container">
                    <table className="trade-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Action</th>
                                <th>Price</th>
                                <th>Shares</th>
                                <th>Profit</th>
                                <th>System</th>
                                <th>Success</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.daily_data.slice(-15).reverse().map((trade, i) => {
                                // Calculate estimated shares (using balance * 0.2 / price as used in live_trader)
                                const estimatedShares = trade.shares || (trade.action === 'BUY' ? (trade.balance * 0.2 / trade.price).toFixed(4) : '-');
                                return (
                                    <tr key={i}>
                                        <td>{trade.date}</td>
                                        <td>
                                            <span className={`badge badge-${trade.action.toLowerCase()}`}>
                                                {trade.action}
                                            </span>
                                        </td>
                                        <td>¥{trade.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                                        <td style={{ color: '#718096' }}>
                                            {trade.action === 'BUY' || trade.action === 'SELL' ? estimatedShares : '-'}
                                        </td>
                                        <td style={{ color: (trade.profit || 0) >= 0 ? '#00f5d4' : '#ff4d4d', fontWeight: 600 }}>
                                            {(trade.profit || 0) >= 0 ? '+' : ''}{(trade.profit || 0).toFixed(2)}
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
