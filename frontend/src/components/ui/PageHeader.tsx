interface PageHeaderProps {
  title: string
  description?: string
  children?: React.ReactNode
}

export default function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <div className="flex items-start justify-between px-8 pt-8 pb-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-text-primary">{title}</h1>
        {description && (
          <p className="text-[13px] text-text-tertiary mt-1">{description}</p>
        )}
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  )
}
