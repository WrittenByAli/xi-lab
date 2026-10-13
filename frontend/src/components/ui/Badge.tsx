import clsx from 'clsx'

type Variant = 'default' | 'bat' | 'bowl' | 'allround' | 'keeper' | 'accent' | 'success' | 'danger'

const styles: Record<Variant, string> = {
  default: 'bg-surface-3 text-text-secondary',
  bat: 'bg-role-bat/12 text-role-bat',
  bowl: 'bg-role-bowl/12 text-role-bowl',
  allround: 'bg-role-allround/12 text-role-allround',
  keeper: 'bg-role-keeper/12 text-role-keeper',
  accent: 'bg-accent/12 text-accent',
  success: 'bg-success/12 text-success',
  danger: 'bg-danger/12 text-danger',
}

interface BadgeProps {
  children: React.ReactNode
  variant?: Variant
  className?: string
}

export default function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold uppercase tracking-[0.04em] leading-none',
        styles[variant],
        className
      )}
    >
      {children}
    </span>
  )
}

export function roleBadgeVariant(role: string): Variant {
  if (role.includes('keeper')) return 'keeper'
  if (role.includes('bat_allrounder') || role.includes('bowl_allrounder')) return 'allround'
  if (role.includes('bowl')) return 'bowl'
  return 'bat'
}

export function roleLabel(role: string): string {
  const map: Record<string, string> = {
    pure_bat: 'Batter',
    pure_bowl: 'Bowler',
    bat_allrounder: 'Bat AR',
    bowl_allrounder: 'Bowl AR',
    keeper_bat: 'WK-Bat',
  }
  return map[role] || role
}
