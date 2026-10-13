import clsx from 'clsx'

export default function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        'animate-pulse rounded-lg bg-surface-3/50',
        className
      )}
    />
  )
}
