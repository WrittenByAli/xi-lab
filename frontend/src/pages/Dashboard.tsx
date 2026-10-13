import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Users, Swords, UserSearch, BarChart3, ArrowRight, CircleCheck, CircleAlert } from 'lucide-react'
import Card, { CardTitle } from '../components/ui/Card'
import StatCard from '../components/ui/StatCard'
import PageHeader from '../components/ui/PageHeader'
import Skeleton from '../components/ui/Skeleton'
import type { BacktestMetrics } from '../lib/api'
import { api } from '../lib/api'

const quickLinks = [
  { to: '/squad', icon: Users, label: 'Build XI', desc: 'Select your best XI from a squad' },
  { to: '/simulate', icon: Swords, label: 'Simulate', desc: 'Head-to-head match simulation' },
  { to: '/players', icon: UserSearch, label: 'Players', desc: 'Explore player profiles & ratings' },
  { to: '/results', icon: BarChart3, label: 'Results', desc: 'Model performance & calibration' },
]

export default function Dashboard() {
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null)
  const [health, setHealth] = useState<{ status: string; version: string } | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([api.getResults(), api.health()]).then(([m, h]) => {
      if (m.status === 'fulfilled') setMetrics(m.value)
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

        {/* Stat cards */}
        <div className="grid grid-cols-4 gap-3">
          {loading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[88px] rounded-xl" />
            ))
          ) : (
            <>
              <StatCard
                label="Test Matches"
                value={metrics?.test_matches?.toLocaleString() ?? '—'}
                sub="held-out"
              />
              <StatCard
                label="Accuracy"
                value={metrics?.accuracy != null ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
                sub={metrics?.accuracy != null ? (metrics.accuracy > 0.55 ? 'above coin flip' : '') : ''}
                trend={metrics?.accuracy != null ? (metrics.accuracy > 0.55 ? 'up' : 'neutral') : undefined}
              />
              <StatCard
                label="Brier Score"
                value={metrics?.brier_score?.toFixed(4) ?? '—'}
                sub="baseline: 0.2500"
              />
              <StatCard
                label="Kelly ROI"
                value={metrics?.kelly_roi != null ? `${(metrics.kelly_roi * 100).toFixed(1)}%` : '—'}
                trend={metrics?.kelly_roi != null ? (metrics.kelly_roi > 0 ? 'up' : 'down') : undefined}
              />
            </>
          )}
        </div>

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

        {/* Calibration preview */}
        {metrics?.calibration && metrics.calibration.length > 0 && (
          <Card>
            <CardTitle>Calibration</CardTitle>
            <div className="mt-3 flex items-end gap-1" style={{ height: 120 }}>
              {metrics.calibration.map((bin) => {
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
