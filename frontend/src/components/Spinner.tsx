export default function Spinner({ label }: { label?: string }) {
  return (
    <span>
      <span className="spinner" />
      {label ?? 'Working…'}
    </span>
  )
}
