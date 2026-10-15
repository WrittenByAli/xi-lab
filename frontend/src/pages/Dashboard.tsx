import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Users, Swords, UserSearch, BarChart3, ArrowRight, CircleCheck, CircleAlert, GitCompareArrows } from 'lucide-react'
import Card, { CardTitle } from '../components/ui/Card'
import PageHeader from '../components/ui/PageHeader'
import Skeleton from '../components/ui/Skeleton'
import type { BacktestMetrics } from '../lib/api'
import { api } from '../lib/api'

const quickLinks = [
  { to: '/squad', icon: Users, label: 'Build XI', desc: 'Select your best XI from a squad' },
  { to: '/simulate', icon: Swords, label: 'Simulate', desc: 'Head-to-head match simulation' },
  { to: '/compare', icon: GitCompareArrows, label: 'Compare', desc: 'Compare XI variants head-to-head' },
  { to: '/players', icon: UserSearch, label: 'Players', desc: 'Explore player profiles & ratings' },
  { to: '/results', icon: BarChart3, label: 'Results', desc: 'Model performance & calibration' },
]

export default function Dashboard() {
  const [t20Metrics, setT20Metrics] = useState<BacktestMetrics | null>(null)
  const [odiMetrics, setOdiMetrics] = useState<BacktestMetrics | null>(null)
  const [health, setHealth] = useState<{ status: string; version: string } | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      api.getResults('t20'),
      api.getResults('odi'),
      api.health(),
    ]).then(([t20, odi, h]) => {
      if (t20.status === 'fulfilled') setT20Metrics(t20.value)
      if (odi.status === 'fulfilled') setOdiMetrics(odi.value)
      if (h.status === 'fulfilled') setHealth(h.value)
      setLoading(false)
    })
  }, [])

  return (
    <div className="pb-12">
      <PageHeader
        title="Dashboard"
        description="XI-Lab — context-aware cricket team selection engine"
      />

      <div className="px-8 space-y-6">
        {/* Health bar */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-[12px]">
            {health ? (
              <>
                <CircleCheck className="w-3.5 h-3.5 text-success" />
                <span className="text-success font-medium">API Online</span>
                <span className="text-text-tertiary">v{health.version}</span>
              </>
            ) : loading ? (
              <Skeleton className="w-24 h-4" />
            ) : (
              <>
                <CircleAlert className="w-3.5 h-3.5 text-danger" />
                <span className="text-danger font-medium">API Offline</span>
              </>
            )}
          </div>
        </div>

        {/* Format comparison cards */}
        {loading ? (
          <div className="grid grid-cols-2 gap-4">
            {[0, 1].map((i) => (
              <Skeleton key={i} className="h-44 rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'T20', metrics: t20Metrics, accent: 'text-accent' },
              { label: 'ODI', metrics: odiMetrics, accent: 'text-role-bowl' },
            ].map(({ label, metrics, accent }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <Card>
                  <div className="flex items-center justify-between mb-3">
                    <span className={`text-[13px] font-semibold uppercase tracking-wider ${accent}`}>{label}</span>
                    <span className="text-[11px] text-text-tertiary">{metrics?.test_matches?.toLocaleString() ?? '—'} test matches</span>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Accuracy</div>
                      <div className="stat text-lg font-semibold text-text-primary mt-0.5">
                        {metrics?.accuracy != null ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Brier</div>
                      <div className="stat text-lg font-semibold text-text-primary mt-0.5">
                        {metrics?.brier_score?.toFixed(4) ?? '—'}
                      </div>
                      {metrics?.brier_score != null && (
                        <div className={`text-[10px] ${metrics.brier_score < 0.25 ? 'text-success' : 'text-danger'}`}>
                          {metrics.brier_score < 0.25 ? 'below' : 'above'} 0.25
                        </div>
                      )}
                    </div>
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-text-tertiary">Kelly ROI</div>
                      <div className="stat text-lg font-semibold text-text-primary mt-0.5">
                        {metrics?.kelly_roi != null ? `${(metrics.kelly_roi * 100).toFixed(1)}%` : '—'}
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        )}

        {/* Quick links */}
        <div>
          <CardTitle className="mb-3">Quick Actions</CardTitle>
          <div className="grid grid-cols-2 gap-3">
            {quickLinks.map(({ to, icon: Icon, label, desc }, i) => (
              <Link key={to} to={to}>
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: i * 0.05 }}
                >
                  <Card hover className="group">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-surface-3 flex items-center justify-center">
                          <Icon className="w-4 h-4 text-text-secondary group-hover:text-accent transition-colors" strokeWidth={1.8} />
                        </div>
                        <div>
                          <div className="text-[13px] font-medium text-text-primary">{label}</div>
                          <div className="text-[12px] text-text-tertiary">{desc}</div>
                        </div>
                      </div>
                      <ArrowRight className="w-4 h-4 text-text-tertiary group-hover:text-accent group-hover:translate-x-0.5 transition-all" />
                    </div>
                  </Card>
                </motion.div>
              </Link>
            ))}
          </div>
        </div>

        {/* Calibration preview — T20 */}
        {t20Metrics?.calibration && t20Metrics.calibration.length > 0 && (
          <Card>
            <div className="flex items-center justify-between mb-3">
              <CardTitle>Calibration</CardTitle>
              <span className="text-[11px] font-medium uppercase tracking-wider text-accent/70">T20</span>
            </div>
            <div className="flex items-end gap-1" style={{ height: 120 }}>
              {t20Metrics.calibration.map((bin) => {
                const h = Math.max(4, bin.actual_mean * 100)
                const expected = bin.predicted_mean * 100
                return (
                  <div key={bin.bin_center} className="flex-1 flex flex-col items-center gap-1">
                    <div className="w-full flex gap-px justify-center" style={{ height: 100 }}>
                      <div
                        className="w-3 rounded-t bg-accent/60 self-end transition-all"
                        style={{ height: `${h}%` }}
                        title={`Actual: ${(bin.actual_mean * 100).toFixed(1)}%`}
                      />
                      <div
                        className="w-3 rounded-t bg-surface-4 self-end transition-all"
                        style={{ height: `${expected}%` }}
                        title={`Predicted: ${(bin.predicted_mean * 100).toFixed(1)}%`}
                      />
                    </div>
                    <span className="stat text-[10px] text-text-tertiary">{(bin.bin_center * 100).toFixed(0)}</span>
                  </div>
                )
              })}
            </div>
            <div className="flex items-center gap-4 mt-3 text-[11px] text-text-tertiary">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-sm bg-accent/60" />
                Actual
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-sm bg-surface-4" />
                Predicted
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}
