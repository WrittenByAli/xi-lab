import { useEffect, useState } from 'react'
import { MapPin, Users2, Sprout, Coins } from 'lucide-react'
import Card, { CardTitle } from '../ui/Card'
import { api } from '../../lib/api'
import type { Gender, PitchCondition, TossDecision, TeamSummary, Venue } from '../../lib/api'

const PITCH_OPTIONS: { value: PitchCondition; label: string; hint: string }[] = [
  { value: 'auto', label: 'Auto (venue history)', hint: "Uses the ground's own learned scoring/wicket tendency — no manual override" },
  { value: 'balanced', label: 'Balanced', hint: 'No pitch adjustment either way' },
  { value: 'batting_friendly', label: 'Batting-friendly', hint: 'Flat deck: fewer dots/wickets, more boundaries' },
  { value: 'bowling_friendly', label: 'Bowling-friendly', hint: 'Juicier deck: more dots/wickets, fewer boundaries' },
  { value: 'spin_friendly', label: 'Spin-friendly (turning)', hint: 'Extra bite for spinners specifically' },
  { value: 'seam_friendly', label: 'Seam-friendly (green)', hint: 'Extra bite for pace bowlers specifically' },
]

const TOSS_OPTIONS: { value: TossDecision | 'auto'; label: string }[] = [
  { value: 'auto', label: 'Undecided' },
  { value: 'bat', label: 'Bat first' },
  { value: 'bowl', label: 'Bowl first' },
]

const selectCls =
  'w-full px-3 py-2 rounded-lg bg-surface-2 border border-surface-border text-[12px] ' +
  'text-text-primary focus:outline-none focus:border-accent/40 cursor-pointer'

interface Props {
  format: string
  gender: Gender
  showOpponent?: boolean
  excludeTeam?: string
  opponentTeam: string | null
  onOpponentChange: (team: string | null) => void
  venue: string | null
  onVenueChange: (venueId: string | null) => void
  pitchCondition: PitchCondition
  onPitchChange: (p: PitchCondition) => void
  tossDecision: TossDecision | null
  onTossChange: (t: TossDecision | null) => void
}

export default function MatchContextPanel({
  format,
  gender,
  showOpponent = true,
  excludeTeam,
  opponentTeam,
  onOpponentChange,
  venue,
  onVenueChange,
  pitchCondition,
  onPitchChange,
  tossDecision,
  onTossChange,
}: Props) {
  const [teams, setTeams] = useState<TeamSummary[]>([])
  const [venues, setVenues] = useState<Venue[]>([])

  useEffect(() => {
    if (!showOpponent) return
    api.listTeams(format, gender, 100).then(setTeams).catch(() => setTeams([]))
  }, [format, gender, showOpponent])

  useEffect(() => {
    api.getVenues(format, gender).then(setVenues).catch(() => setVenues([]))
  }, [format, gender])

  const selectedPitch = PITCH_OPTIONS.find((p) => p.value === pitchCondition)

  return (
    <Card padding="sm">
      <CardTitle className="px-2 pt-1">Match Context</CardTitle>
      <div
        className={`grid gap-3 p-2 mt-2 ${showOpponent ? 'grid-cols-4' : 'grid-cols-3'}`}
      >
        {showOpponent && (
          <div>
            <label className="flex items-center gap-1.5 text-[11px] text-text-tertiary mb-1.5">
              <Users2 className="w-3 h-3" /> Opponent
            </label>
            <select
              className={selectCls}
              value={opponentTeam ?? ''}
              onChange={(e) => onOpponentChange(e.target.value || null)}
            >
              <option value="">None (rating-based)</option>
              {teams
                .filter((t) => t.team !== excludeTeam)
                .map((t) => (
                  <option key={t.team} value={t.team}>
                    {t.team}
                  </option>
                ))}
            </select>
          </div>
        )}

        <div>
          <label className="flex items-center gap-1.5 text-[11px] text-text-tertiary mb-1.5">
            <MapPin className="w-3 h-3" /> Ground
          </label>
          <select
            className={selectCls}
            value={venue ?? ''}
            onChange={(e) => onVenueChange(e.target.value || null)}
          >
            <option value="">Any venue</option>
            {venues.map((v) => (
              <option key={v.venue_id} value={v.venue_id}>
                {v.name} ({v.match_count})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="flex items-center gap-1.5 text-[11px] text-text-tertiary mb-1.5">
            <Sprout className="w-3 h-3" /> Pitch condition
          </label>
          <select
            className={selectCls}
            value={pitchCondition}
            onChange={(e) => onPitchChange(e.target.value as PitchCondition)}
          >
            {PITCH_OPTIONS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
          {selectedPitch && (
            <p className="text-[10px] text-text-tertiary/70 mt-1 leading-snug">{selectedPitch.hint}</p>
          )}
        </div>

        <div>
          <label className="flex items-center gap-1.5 text-[11px] text-text-tertiary mb-1.5">
            <Coins className="w-3 h-3" /> Toss
          </label>
          <select
            className={selectCls}
            value={tossDecision ?? 'auto'}
            onChange={(e) =>
              onTossChange(e.target.value === 'auto' ? null : (e.target.value as TossDecision))
            }
          >
            {TOSS_OPTIONS.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </Card>
  )
}
