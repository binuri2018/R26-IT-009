import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend
} from 'recharts';

const EMOTION_VALUES = {
  happy: 100,
  surprised: 80,
  neutral: 60,
  confused: 30,
  frustrated: 10,
  unknown: 50
};

const ResearchCharts = ({ responses }) => {
  if (!responses || responses.length === 0) return null;

  // 1. Process Timeline Data
  const timelineData = responses.map((r, idx) => ({
    name: `Q${idx + 1}`,
    emotion: r.detectedExpression,
    valence: EMOTION_VALUES[r.detectedExpression] || 60,
    time: r.responseTime,
    correct: r.correctness === 1,
    changes: r.answerChanges
  }));

  // 2. Process Load Distribution
  const loadData = responses.map((r, idx) => ({
    name: `Q${idx + 1}`,
    time: r.responseTime,
    changes: r.answerChanges,
    load: (r.responseTime * 0.5) + (r.answerChanges * 10) + (r.correctness === 0 ? 20 : 0)
  }));

  return (
    <div className="research-charts-container">
      
      {/* 1. Engagement Timeline */}
      <div className="report-section" style={{ marginTop: '24px' }}>
        <h3 className="section-title">📉 Behavioral Engagement Timeline</h3>
        <p style={{ fontSize: '12px', color: '#6b7280', marginBottom: '16px' }}>
          Visualizing emotional valence shifts across the assessment session.
        </p>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer>
            <AreaChart data={timelineData}>
              <defs>
                <linearGradient id="colorValence" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0E7C66" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#0E7C66" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" vertical={false} />
              <XAxis dataKey="name" stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis hide domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ background: '#ffffff', border: '1px solid #e5e9ef', borderRadius: '8px' }}
                itemStyle={{ color: '#111827' }}
                labelStyle={{ color: '#6b7280' }}
                formatter={(value, name, props) => [(props.payload.emotion || 'unknown').toUpperCase(), 'Emotion']}
              />
              <Area 
                type="monotone" 
                dataKey="valence" 
                stroke="#0E7C66" 
                strokeWidth={3}
                fillOpacity={1} 
                fill="url(#colorValence)" 
                animationDuration={1500}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. Cognitive Effort Analysis */}
      <div className="report-section" style={{ marginTop: '24px' }}>
        <h3 className="section-title">📊 Cognitive Effort Analysis</h3>
        <p style={{ fontSize: '12px', color: '#6b7280', marginBottom: '16px' }}>
          Mapping response time vs. answer revisions to estimate mental load.
        </p>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer>
            <BarChart data={loadData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e9ef" vertical={false} />
              <XAxis dataKey="name" stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ background: '#ffffff', border: '1px solid #e5e9ef', borderRadius: '8px' }}
                cursor={{ fill: 'rgba(15,23,42,0.05)' }}
              />
              <Legend verticalAlign="top" height={36}/>
              <Bar name="Time (s)" dataKey="time" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar name="Answer Changes" dataKey="changes" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};

export default ResearchCharts;
