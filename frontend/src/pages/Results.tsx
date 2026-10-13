import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts'
import PageHeader from '../components/ui/PageHeader'
import Card, { CardTitle } from '../components/ui/Card'
import StatCard from '../components/ui/StatCard'
import Skeleton from '../components/ui/Skeleton'
import { api } from '../lib/api'
import type { BacktestMetrics } from '../lib/api'

export default function Results() {
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null)
  const [allMetrics, setAllMetrics] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([api.getResults(), api.getAllMetrics()]).then(([m, a]) => {
      if (m.status === 'fulfilled') setMetrics(m.value)
      if (a.status === 'fulfilled') setAllMetrics(a.value)
      setLoading(false)
    })
  }, [])

  const calibrationData = metrics?.calibration?.map((c) => ({
    predicted: +(c.predicted_mean * 100).toFixed(1),
    actual: +(c.actual_mean * 100).toFixed(1),
    n: c.count,
  })) ?? []

  return (
    <div className="pb-12">
      <PageHeader
        title="Model Results"
        description="Backtest performance, calibration, and model evaluation metrics"
      />

      <div className="px-8 space-y-6">
        {loading ? (
          <>
            <div className="grid grid-cols-4 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-[88px] rounded-xl" />
              ))}
            </div>
            <Skeleton className="h-80 rounded-xl" />
          </>
        ) : (
          <>
            {/* Key metrics */}
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-4 gap-3"
            >
              <StatCard
                label="Test Matches"
                value={metrics?.test_matches?.toLocaleString() ?? '—'}
              />
              <StatCard
                label="Accuracy"
                value={metrics?.accuracy != null ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
                sub="vs 50% baseline"
                trend={metrics?.accuracy != null && metrics.accuracy > 0.55 ? 'up' : 'neutral'}
              />
              <StatCard
                label="Brier Score"
                value={metrics?.brier_score?.toFixed(4) ?? '—'}
                sub="↓ better"
              />
              <StatCard
                label="Log Loss"
                value={metrics?.log_loss?.toFixed(4) ?? '—'}
                sub="↓ better"
              />
            </motion.div>

            {/* Calibration chart */}
            {calibrationData.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card>
                  <CardTitle>Calibration Plot</CardTitle>
                  <p className="text-[12px] text-text-tertiary mt-1 mb-4">
                    Perfect calibration follows the diagonal — if the model predicts 70% WP, the team should win 70% of the time.
                  </p>
                  <div style={{ height: 320 }}>
                    <ResponsiveContainer>
                      <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid
                          stroke="#2e2e38"
                          strokeDasharray="3 3"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="predicted"
                          name="Predicted"
                          unit="%"
                          type="number"
                          domain={[0, 100]}
                          tick={{ fill: '#6b6966', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                          axisLine={{ stroke: '#2e2e38' }}
                          tickLine={false}
                          label={{ value: 'Predicted WP', position: 'bottom', offset: 5, fill: '#6b6966', fontSize: 11 }}
                        />
                        <YAxis
                          dataKey="actual"
                          name="Actual"
                          unit="%"
                          type="number"
                          domain={[0, 100]}
                          tick={{ fill: '#6b6966', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                          axisLine={{ stroke: '#2e2e38' }}
                          tickLine={false}
                          label={{ value: 'Actual Win %', angle: -90, position: 'left', offset: 5, fill: '#6b6966', fontSize: 11 }}
                        />
                        <Tooltip
                          cursor={false}
                          contentStyle={{
                            background: '#1a1a1f',
                            border: '1px solid #2e2e38',
                            borderRadius: 10,
                            fontSize: 12,
                            fontFamily: 'JetBrains Mono',
                            color: '#e8e6e3',
                          }}
                          formatter={(value) => [`${value}%`]}
                          labelFormatter={() => ''}
                        />
                        <ReferenceLine
                          segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
                          stroke="#e5a84b"
                          strokeWidth={1}
                          strokeDasharray="6 4"
                          strokeOpacity={0.4}
                        />
                        <Scatter
                          data={calibrationData}
                          fill="#e5a84b"
                          fillOpacity={0.8}
                          r={6}
                          shape="circle"
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </motion.div>
            )}

            {/* Model details */}
            <div className="grid grid-cols-2 gap-4">
              {/* Ability model */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card>
                  <CardTitle>Ability Model</CardTitle>
                  {allMetrics?.ability ? (
                    <div className="mt-3 space-y-2">
                      {Object.entries(allMetrics.ability).map(([k, v]) => (
                        <div key={k} className="flex justify-between items-center py-1 border-b border-surface-border/30 last:border-0">
                          <span className="text-[12px] text-text-tertiary">{k.replace(/_/g, ' ')}</span>
                          <span className="stat text-[12px] text-text-primary">
                            {typeof v === 'number' ? v.toFixed(4) : String(v)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[12px] text-text-tertiary mt-2">Not available — run training pipeline</p>
                  )}
                </Card>
              </motion.div>

              {/* Ball model */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
              >
                <Card>
                  <CardTitle>Ball Outcome Model</CardTitle>
                  {allMetrics?.ball ? (
                    <div className="mt-3 space-y-2">
                      {['val_log_loss', 'val_accuracy', 'test_log_loss', 'test_accuracy'].map((k) => {
                        const v = (allMetrics.ball as Record<string, any>)[k]
                        if (v === undefined) return null
                        return (
                          <div key={k} className="flex justify-between items-center py-1 border-b border-surface-border/30 last:border-0">
                            <span className="text-[12px] text-text-tertiary">{k.replace(/_/g, ' ')}</span>
                            <span className="stat text-[12px] text-text-primary">
                              {typeof v === 'number' ? v.toFixed(4) : String(v)}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <p className="text-[12px] text-text-tertiary mt-2">Not available — run training pipeline</p>
                  )}
                </Card>
              </motion.div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
