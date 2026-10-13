import { motion } from 'framer-motion'
import Badge, { roleBadgeVariant, roleLabel } from '../ui/Badge'
import type { PlayerProfile, PlayerSummary } from '../../lib/api'

type Player = PlayerProfile | PlayerSummary

interface PlayerCardProps {
  player: Player
  rank?: number
  compact?: boolean
  onClick?: () => void
}

export default function PlayerCard({ player, rank, compact, onClick }: PlayerCardProps) {
  const isProfile = 'bat_sr' in player
  const p = player as PlayerProfile

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: rank ? rank * 0.03 : 0 }}
      onClick={onClick}
      className={`
        group relative rounded-xl border border-surface-border bg-surface-1
        transition-all duration-150
        ${onClick ? 'cursor-pointer hover:border-surface-4 hover:bg-surface-2/60' : ''}
        ${compact ? 'p-3' : 'p-4'}
      `}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          {/* Rank indicator */}
          {rank !== undefined && (
            <div className="w-6 h-6 rounded-md bg-surface-3 flex items-center justify-center shrink-0">
              <span className="stat text-[11px] font-bold text-text-tertiary">{rank}</span>
            </div>
          )}

          {/* Player initial avatar */}
          <div className="w-9 h-9 rounded-lg bg-surface-3 flex items-center justify-center shrink-0">
            <span className="text-[14px] font-bold text-text-secondary">
              {player.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
            </span>
          </div>

          <div className="min-w-0">
            <div className="text-[13px] font-medium text-text-primary truncate leading-tight">
              {player.name}
            </div>
            <div className="flex items-center gap-1.5 mt-1">
              <Badge variant={roleBadgeVariant(player.role)}>{roleLabel(player.role)}</Badge>
              {player.is_keeper && <Badge variant="keeper">WK</Badge>}
            </div>
          </div>
        </div>

        {/* Rating pill */}
        <div className="flex gap-2 shrink-0">
          {player.bat_rating != null && (
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-text-tertiary">BAT</div>
              <div className="stat text-[13px] font-semibold text-role-bat">
                {typeof player.bat_rating === 'number' ? player.bat_rating.toFixed(1) : '—'}
              </div>
            </div>
          )}
          {player.bowl_rating != null && (
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-text-tertiary">BOWL</div>
              <div className="stat text-[13px] font-semibold text-role-bowl">
                {typeof player.bowl_rating === 'number' ? Math.abs(player.bowl_rating).toFixed(1) : '—'}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Extended stats */}
      {isProfile && !compact && (
        <div className="grid grid-cols-4 gap-px mt-3.5 rounded-lg overflow-hidden bg-surface-border/50">
          {[
            { label: 'SR', value: p.bat_sr, fmt: (v: number) => v.toFixed(1) },
            { label: 'AVG', value: p.bat_avg, fmt: (v: number) => v.toFixed(1) },
            { label: 'ECON', value: p.bowl_econ, fmt: (v: number) => v.toFixed(2) },
            { label: 'B SR', value: p.bowl_sr, fmt: (v: number) => v.toFixed(1) },
          ].map(({ label, value, fmt }) => (
            <div key={label} className="bg-surface-2 px-2.5 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wider text-text-tertiary">{label}</div>
              <div className="stat text-[13px] font-medium text-text-primary mt-0.5">
                {value != null ? fmt(value) : '—'}
              </div>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}
