import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Zap, GripVertical } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'
import Card, { CardTitle } from '../components/ui/Card'
import Badge, { roleBadgeVariant, roleLabel } from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Skeleton from '../components/ui/Skeleton'
import PlayerCard from '../components/cricket/PlayerCard'
import { api } from '../lib/api'
import type { PlayerIn, PlayerSearchResult, RecommendResponse } from '../lib/api'

function makePlayerIn(p: PlayerSearchResult): PlayerIn {
  return {
    player_id: p.player_id,
    name: p.name,
    role: 'pure_bat',
    bat_rating: 0.5,
    bowl_rating: 0.5,
    is_keeper: p.is_keeper,
    is_overseas: false,
    bat_hand: p.bat_hand ?? 'R',
    bowl_type: p.bowl_type ?? 'none',
  }
}

export default function SquadBuilder() {
  const [squad, setSquad] = useState<PlayerIn[]>([])
  const [searchQ, setSearchQ] = useState('')
  const [searchResults, setSearchResults] = useState<PlayerSearchResult[]>([])
  const [result, setResult] = useState<RecommendResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const debounce = useRef<ReturnType<typeof setTimeout>>(undefined)

  const searchPlayers = useCallback((q: string) => {
    setSearchQ(q)
    if (debounce.current) clearTimeout(debounce.current)
    if (q.length < 2) { setSearchResults([]); return }
    debounce.current = setTimeout(async () => {
      try {
        const r = await api.searchPlayers(q)
        setSearchResults(r.filter(p => !squad.some(s => s.player_id === p.player_id)))
      } catch (_) { setSearchResults([]) }
    }, 200)
  }, [squad])

  const addPlayer = (p: PlayerSearchResult) => {
    if (squad.some(s => s.player_id === p.player_id)) return
    setSquad(prev => [...prev, makePlayerIn(p)])
    setSearchQ('')
    setSearchResults([])
  }

  const removePlayer = (id: string) => {
    setSquad(prev => prev.filter(p => p.player_id !== id))
  }

  const recommend = async () => {
    if (squad.length < 11) {
      setError('Need at least 11 players in the squad')
      return
    }
    setError('')
    setLoading(true)
    try {
      const r = await api.recommend(squad)
      setResult(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get recommendation')
    }
    setLoading(false)
  }

  return (
    <div className="pb-12">
      <PageHeader
        title="Squad Builder"
        description="Add players to your squad, then get the optimal XI"
      >
        <Button onClick={recommend} disabled={squad.length < 11 || loading}>
          <Zap className="w-4 h-4" />
          {loading ? 'Computing...' : 'Get Best XI'}
        </Button>
      </PageHeader>

      <div className="px-8 flex gap-6">
        {/* Left: Squad management */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Search to add */}
          <div className="relative">
            <Input
              icon
              placeholder="Search players to add..."
              value={searchQ}
              onChange={(e) => searchPlayers(e.target.value)}
            />
            <AnimatePresence>
              {searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="absolute z-20 top-full mt-1 w-full bg-surface-2 border border-surface-border rounded-xl shadow-2xl shadow-black/40 overflow-hidden max-h-64 overflow-y-auto"
                >
                  {searchResults.slice(0, 8).map((p) => (
                    <div
                      key={p.player_id}
                      onClick={() => addPlayer(p)}
                      className="flex items-center gap-3 px-4 py-2.5 cursor-pointer hover:bg-surface-3 transition-colors"
                    >
                      <Plus className="w-3.5 h-3.5 text-accent shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="text-[13px] text-text-primary truncate">{p.name}</div>
                        <div className="text-[11px] text-text-tertiary">
                          {p.bat_hand}HB{p.bowl_type && p.bowl_type !== 'none' ? ` · ${p.bowl_type}` : ''}
                          {p.is_keeper ? ' · WK' : ''}
                        </div>
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Squad list */}
          <Card padding="sm">
            <div className="flex items-center justify-between px-2 py-1 mb-2">
              <CardTitle>
                Squad ({squad.length})
              </CardTitle>
              {squad.length > 0 && (
                <button
                  onClick={() => { setSquad([]); setResult(null) }}
                  className="text-[11px] text-text-tertiary hover:text-danger transition-colors cursor-pointer"
                >
                  Clear all
                </button>
              )}
            </div>

            {squad.length === 0 ? (
              <div className="text-center py-12 text-[13px] text-text-tertiary">
                Search and add players to build your squad
              </div>
            ) : (
              <div className="space-y-1">
                <AnimatePresence mode="popLayout">
                  {squad.map((p) => (
                    <motion.div
                      key={p.player_id}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      layout
                      className="overflow-hidden"
                    >
                      <div className="flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-surface-2/60 group">
                        <GripVertical className="w-3.5 h-3.5 text-text-tertiary/40 shrink-0" />
                        <div className="w-7 h-7 rounded-md bg-surface-3 flex items-center justify-center shrink-0">
                          <span className="text-[10px] font-bold text-text-secondary">
                            {p.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
                          </span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <span className="text-[13px] font-medium text-text-primary truncate block">{p.name}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          {p.is_keeper && <Badge variant="keeper">WK</Badge>}
                          <span className="text-[11px] text-text-tertiary">{p.bat_hand}HB</span>
                        </div>
                        <button
                          onClick={() => removePlayer(p.player_id)}
                          className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-danger/10 transition-all cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5 text-danger/60" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}
          </Card>

          {error && (
            <div className="text-[12px] text-danger bg-danger/8 border border-danger/20 rounded-lg px-4 py-2.5">
              {error}
            </div>
          )}
        </div>

        {/* Right: Result panel */}
        <div className="w-[380px] shrink-0 space-y-4">
          {loading ? (
            <>
              <Skeleton className="h-20 rounded-xl" />
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-16 rounded-xl" />
              ))}
            </>
          ) : result ? (
            <motion.div
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-4"
            >
              {/* WP header */}
              <Card className="bg-accent/5 border-accent/15">
                <div className="text-center">
                  <div className="text-[11px] uppercase tracking-wider text-accent/70">Win Probability</div>
                  <div className="stat text-3xl font-bold text-accent mt-1">
                    {(result.win_prob * (result.win_prob > 1 ? 1 : 100)).toFixed(1)}
                    <span className="text-lg ml-0.5">%</span>
                  </div>
                </div>
              </Card>

              {/* Selected XI */}
              <div>
                <CardTitle className="mb-3">Selected XI</CardTitle>
                <div className="space-y-2">
                  {result.xi.map((p, i) => (
                    <PlayerCard key={p.player_id} player={p} rank={i + 1} compact />
                  ))}
                </div>
              </div>

              {/* Batting order */}
              <Card>
                <CardTitle>Batting Order</CardTitle>
                <div className="mt-3 space-y-1">
                  {result.batting_order.map((pid, i) => {
                    const p = result.xi.find(x => x.player_id === pid)
                    return (
                      <div key={pid} className="flex items-center gap-2.5 py-1">
                        <span className="stat text-[11px] text-text-tertiary w-4 text-right">{i + 1}</span>
                        <span className="text-[13px] text-text-primary">{p?.name ?? pid}</span>
                        {p && <Badge variant={roleBadgeVariant(p.role)} className="ml-auto">{roleLabel(p.role)}</Badge>}
                      </div>
                    )
                  })}
                </div>
              </Card>

              {/* Bowling plan */}
              <Card>
                <CardTitle>Bowling Plan</CardTitle>
                <div className="mt-3 space-y-1.5">
                  {Object.entries(result.bowling_plan).map(([pid, overs]) => {
                    const p = result.xi.find(x => x.player_id === pid)
                    return (
                      <div key={pid} className="flex items-center justify-between">
                        <span className="text-[13px] text-text-primary">{p?.name ?? pid}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 rounded-full bg-surface-3 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-role-bowl/60"
                              style={{ width: `${(overs / 4) * 100}%` }}
                            />
                          </div>
                          <span className="stat text-[12px] text-text-secondary w-8 text-right">{overs}ov</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </Card>
            </motion.div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-12 h-12 rounded-xl bg-surface-2 flex items-center justify-center mb-3">
                <Zap className="w-5 h-5 text-text-tertiary" />
              </div>
              <div className="text-[13px] text-text-tertiary">
                Add 11+ players and hit<br />"Get Best XI" to optimize
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
