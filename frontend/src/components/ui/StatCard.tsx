import clsx from 'clsx'

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  trend?: 'up' | 'down' | 'neutral'
  className?: string
}

export default function StatCard({ label, value, sub, trend, className }: StatCardProps) {
  return (
    <div
      className={clsx(
        'rounded-xl border border-surface-border bg-surface-1 p-4 flex flex-col gap-1',
        className
      )}
    >
      <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-text-tertiary">
        {label}
      </span>
      <div className="flex items-baseline gap-2">
        <span className="stat text-2xl font-semibold text-text-primary leading-none">
          {value}
        </span>
        {sub && (
          <span
            className={clsx(
              'text-[12px] font-medium',
              trend === 'up' && 'text-success',
              trend === 'down' && 'text-danger',
              (!trend || trend === 'neutral') && 'text-text-tertiary'
            )}
          >
            {sub}
          </span>
        )}
      </div>
    </div>
  )
}
