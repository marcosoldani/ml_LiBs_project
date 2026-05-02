type Item = { label: string; value: string | number }

export default function MetricGrid({
  items,
}: {
  items: (Item | [string, string | number])[]
}) {
  const normalized: Item[] = items.map((it) =>
    Array.isArray(it) ? { label: it[0], value: it[1] } : it,
  )
  return (
    <div className="metric-grid">
      {normalized.map((m, i) => (
        <div className="metric" key={i}>
          <div className="label">{m.label}</div>
          <div className="value">
            {typeof m.value === 'number'
              ? Number.isInteger(m.value)
                ? m.value.toLocaleString()
                : m.value.toFixed(4)
              : m.value}
          </div>
        </div>
      ))}
    </div>
  )
}
