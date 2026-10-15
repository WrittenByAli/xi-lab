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

const FORMATS = ['t20', 'odi', 'test'] as const

export default function Results() {
  const [format, setFormat] = useState<string>('t20')
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null)
  const [allMetrics, setAllMetrics] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.allSettled([api.getResults(format), api.getAllMetrics()]).then(([m, a]) => {
      if (m.status === 'fulfilled') setMetrics(m.value)
      if (a.status === 'fulfilled') setAllMetrics(a.value)
      setLoading(false)
    })
  }, [format])

  const calibrationData = metrics?.calibration?.map((c) => ({
    predicted: +(c.predicted_mean * 100).toFixed(1),
    actual: +(c.actual_mean * 100).toFixed(1),
    n: c.count,
  })) ?? []

  const brierDelta = metrics?.brier_score != null ? (0.25 - metrics.brier_score) : null
  const logLossDelta = metrics?.log_loss != null ? (0.6931 - metrics.log_loss) : null

  return (
    <div className="pb-12">
      <PageHeader
        title="Model Results"
        description="Backtest performance, calibration, and model evaluation metrics"
      >
        <div className="flex bg-surface-2 rounded-lg p-0.5">
          {FORMATS.map((f) => (
            <button
              key={f}
              onClick={() => setFormat(f)}
              className={`
                px-3 py-1.5 rounded-md text-[12px] font-medium uppercase tracking-wider
                transition-all duration-150 cursor-pointer
                ${format === f
                  ? 'bg-surface-3 text-accent shadow-sm'
                  : 'text-text-tertiary hover:text-text-secondary'}
              `}
            >
              {f}
            </button>
          ))}
        </div>
      </PageHeader>

      <div className="px-8 space-y-6">
        {loading ? (
          <>
            <div className="grid grid-cols-5 gap-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-[88px] rounded-xl" />
              ))}
            </div>
            <Skeleton className="h-80 rounded-xl" />
          </>
        ) : (
          <>
            {/* Key metrics */}
            <motion.div
              key={format}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="grid grid-cols-5 gap-3"
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
                sub={brierDelta != null ? `${brierDelta > 0 ? '+' : ''}${brierDelta.toFixed(4)} vs baseline` : ''}
                trend={brierDelta != null && brierDelta > 0 ? 'up' : 'neutral'}
              />
              <StatCard
                label="Log Loss"
                value={metrics?.log_loss?.toFixed(4) ?? '—'}
                sub={logLossDelta != null ? `${logLossDelta > 0 ? '+' : ''}${logLossDelta.toFixed(4)} vs baseline` : ''}
                trend={logLossDelta != null && logLossDelta > 0 ? 'up' : 'neutral'}
              />
              <StatCard
                label="Kelly ROI"
                value={metrics?.kelly_roi != null ? `${(metrics.kelly_roi * 100).toFixed(2)}%` : '—'}
                sub="simulated edge"
                trend={metrics?.kelly_roi != null && metrics.kelly_roi > 0 ? 'up' : 'neutral'}
              />
            </motion.div>

            {/* Calibration chart */}
            {calibrationData.length > 0 && (
              <motion.div
                key={`cal-${format}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card>
                  <div className="flex items-center justify-between mb-1">
                    <CardTitle>Calibration Plot</CardTitle>
                    <span className="text-[11px] font-medium uppercase tracking-wider text-accent/70">{format}</span>
                  </div>
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
                      {Object.entries(allMetrics.ability)
                        .filter(([, v]) => typeof v === 'number' || typeof v === 'string')
                        .map(([k, v]) => (
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
                      {(() => {
                        const ball = allMetrics.ball as Record<string, any>
                        const valMetrics = ball.val ?? {}
                        const testMetrics = ball.test ?? {}
                        return [
                          { k: 'Val log loss', v: valMetrics.log_loss },
                          { k: 'Val accuracy', v: valMetrics.accuracy },
                          { k: 'Test log loss', v: testMetrics.log_loss },
                          { k: 'Test accuracy', v: testMetrics.accuracy },
                          { k: 'Test samples', v: testMetrics.n_samples },
                        ].filter(({ v }) => v !== undefined).map(({ k, v }) => (
                          <div key={k} className="flex justify-between items-center py-1 border-b border-surface-border/30 last:border-0">
                            <span className="text-[12px] text-text-tertiary">{k}</span>
                            <span className="stat text-[12px] text-text-primary">
                              {typeof v === 'number' ? (v > 100 ? v.toLocaleString() : v.toFixed(4)) : String(v)}
                            </span>
                          </div>
                        ))
                      })()}
                    </div>
                  ) : (
                    <p className="text-[12px] text-text-tertiary mt-2">Not available — run training pipeline</p>
                  )}
                </Card>
              </motion.div>
            </div>

            {/* Feature importance */}
            {allMetrics?.ball && (() => {
              const ball = allMetrics.ball as Record<string, any>
              const imp = (ball.test ?? ball.val)?.feature_importance_top20 as { feature: string; gain: number }[] | undefined
              if (!imp || imp.length === 0) return null
              const maxGain = imp[0].gain
              return (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <Card>
                    <CardTitle>Feature Importance (Ball Model)</CardTitle>
                    <p className="text-[12px] text-text-tertiary mt-1 mb-4">
                      Top features by total gain — how much each feature reduces prediction error.
                    </p>
                    <div className="space-y-1.5">
                      {imp.slice(0, 12).map((f, i) => (
                        <div key={f.feature} className="flex items-center gap-3">
                          <span className="stat text-[11px] text-text-tertiary w-4 text-right">{i + 1}</span>
                          <span className="text-[12px] text-text-secondary w-40 truncate">{f.feature.replace(/_/g, ' ')}</span>
                          <div className="flex-1 h-2 rounded-full bg-surface-3 overflow-hidden">
                            <motion.div
                              className="h-full rounded-full bg-accent/70"
                              initial={{ width: 0 }}
                              animate={{ width: `${(f.gain / maxGain) * 100}%` }}
                              transition={{ duration: 0.4, delay: i * 0.03 }}
                            />
                          </div>
                          <span className="stat text-[11px] text-text-tertiary w-16 text-right">{f.gain.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </motion.div>
              )
            })()}
          </>
        )}
      </div>
    </div>
  )
}
