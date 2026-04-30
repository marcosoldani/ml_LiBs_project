import { useEffect, useState } from 'react'
import { fetchAgingEvolution, fetchOptions, runTask3 } from '../api'
import type {
  AgingEvolutionResponse,
  DatasetOptions,
  Task3Response,
} from '../types'
import Plot, { baseConfig, usePlotLayout } from '../components/Plot'
import MetricGrid from '../components/MetricGrid'
import MetricBar from '../components/MetricBar'
import MetricsTable from '../components/MetricsTable'
import Segmented from '../components/Segmented'
import Spinner from '../components/Spinner'
import { useT } from '../i18n'

type Mode = 'manual' | 'random'

export default function Task3() {
  const t = useT()
  const baseLayout = usePlotLayout()
  const [opts, setOpts] = useState<DatasetOptions | null>(null)
  const [mode, setMode] = useState<Mode>('manual')
  const [aging, setAging] = useState<number>(2)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Task3Response | null>(null)
  const [evolution, setEvolution] = useState<AgingEvolutionResponse | null>(null)
  const [evoSoc, setEvoSoc] = useState<number>(2)
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState<number | null>(null)

  useEffect(() => { fetchOptions().then(setOpts) }, [])

  function reroll() {
    if (!opts) return
    setAging(opts.agings[Math.floor(Math.random() * opts.agings.length)])
  }
  useEffect(() => {
    if (mode === 'random' && opts) reroll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, opts])

  async function run() {
    setRunning(true); setError(null); setResult(null)
    const t0 = performance.now()
    try {
      const data = await runTask3(aging); setResult(data)
      setElapsed((performance.now() - t0) / 1000)
      const evo = await fetchAgingEvolution(evoSoc, aging); setEvolution(evo)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || t.common.runFailed)
    } finally { setRunning(false) }
  }

  useEffect(() => {
    if (result) fetchAgingEvolution(evoSoc, result.excluded_aging).then(setEvolution)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evoSoc])

  const modeOptions = [t.common.manual, t.common.random] as const

  return (
    <>
      <h1 className="page-title">{t.task3.title}</h1>
      <div className="page-caption">{t.task3.caption}</div>
      <div className="banner">{t.task3.banner}</div>

      <div className="card">
        <div className="card-title">{t.common.configuration}</div>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <Segmented
            value={mode === 'manual' ? t.common.manual : t.common.random}
            options={modeOptions}
            onChange={(v) => setMode(v === t.common.manual ? 'manual' : 'random')}
          />
        </div>
        {mode === 'manual' ? (
          <label className="field" style={{ maxWidth: 220 }}>
            {t.task3.agingExclude}
            <select value={aging} onChange={(e) => setAging(Number(e.target.value))}>
              {opts?.agings.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </label>
        ) : (
          <div className="row">
            <span className="pick-chip">Aging <b>{aging}</b></span>
            <button onClick={reroll}>{t.common.reroll}</button>
          </div>
        )}
        <div className="row end" style={{ marginTop: '0.9rem' }}>
          <button className="primary" onClick={run} disabled={running}>
            {running ? <Spinner label={t.task3.spinnerLabel} /> : t.task3.runBtn}
          </button>
        </div>
        {error && <div className="banner banner-error" style={{ marginTop: '0.7rem' }}>{error}</div>}
        {result && elapsed && (
          <div className="banner banner-success" style={{ marginTop: '0.7rem' }}>
            {t.common.pipelineDone} {elapsed.toFixed(1)}{t.common.seconds}.
          </div>
        )}
      </div>

      {!result && !running && (
        <div className="card prose" style={{ marginTop: '1rem' }}>
          <h3>{t.task3.explainHeader}</h3>
          <ol>{t.task3.explain.map((s, i) => <li key={i}>{s}</li>)}</ol>
          <blockquote>{t.task3.why}</blockquote>
        </div>
      )}

      {result && (
        <>
          <h3 style={{ marginTop: '1.5rem' }}>{t.task3.bestModel}</h3>
          <div className="card-caption">{t.task3.excludedAging} = {result.excluded_aging}</div>
          <MetricGrid
            items={[
              [t.common.bestModel, result.best_model_name],
              ['R²', result.metrics_per_model[result.best_model_name].R2],
              ['R² Re(Z)', result.metrics_per_model[result.best_model_name].R2_real],
              ['R² Im(Z)', result.metrics_per_model[result.best_model_name].R2_imag],
              ['MSE', result.metrics_per_model[result.best_model_name].MSE],
              ['MAE', result.metrics_per_model[result.best_model_name].MAE],
              [t.common.trainingTime, `${result.training_time_s[result.best_model_name].toFixed(2)}s`],
              [t.common.testPoints, result.n_test_points],
            ]}
          />

          <hr />
          <h3>{t.common.benchmark}</h3>
          <div className="cols-2">
            <MetricBar metrics={result.metrics_per_model} metricKey="R2" title="R²" />
            <MetricBar metrics={result.metrics_per_model} metricKey="MSE" title="MSE" />
          </div>
          <div style={{ marginTop: '0.8rem' }}>
            <MetricsTable
              metrics={result.metrics_per_model}
              trainingTime={result.training_time_s}
              sortKey="R2" highlight={result.best_model_name}
            />
          </div>

          <hr />
          <h3>{t.task3.evolutionTitle}</h3>
          <label className="field" style={{ maxWidth: 220 }}>
            {t.task3.evoLabel}
            <select value={evoSoc} onChange={(e) => setEvoSoc(Number(e.target.value))}>
              {opts?.socs.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          {evolution && (
            <Plot
              data={evolution.panels.flatMap((panel, idx) =>
                panel.traces.map((tr) => ({
                  type: 'scatter' as const, mode: 'lines+markers' as const,
                  x: tr.z_real, y: tr.z_imag_neg,
                  xaxis: idx === 0 ? 'x' : `x${idx + 1}`,
                  yaxis: idx === 0 ? 'y' : `y${idx + 1}`,
                  name: tr.label, legendgroup: String(tr.aging),
                  showlegend: idx === 0,
                  line: { color: tr.color, width: tr.dashed ? 2.5 : 1.8, dash: tr.dashed ? 'dash' : 'solid' },
                  marker: { size: 5 },
                })),
              )}
              layout={{
                ...baseLayout,
                title: t.task3.evolutionTitle.replace(/^.{2}\s/, '') + ` (SOC=${evolution.soc})`,
                grid: { rows: 1, columns: evolution.panels.length, pattern: 'independent' },
                ...Object.fromEntries(
                  evolution.panels.flatMap((_, i) => [
                    [`xaxis${i === 0 ? '' : i + 1}`, { title: 'Re(Z) [mΩ]' }],
                    [`yaxis${i === 0 ? '' : i + 1}`, { title: i === 0 ? '-Im(Z) [mΩ]' : '' }],
                  ]),
                ),
                annotations: evolution.panels.map((p, i) => ({
                  text: `T=${p.temperature}°C`, showarrow: false, x: 0.5, y: 1.07,
                  xref: i === 0 ? 'x domain' : `x${i + 1} domain`,
                  yref: i === 0 ? 'y domain' : `y${i + 1} domain`,
                  font: { size: 12 },
                })),
                height: 420,
              }}
              config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
            />
          )}

          <hr />
          <h3>{t.task3.perTempTitle}</h3>
          <div className="cols-2">
            <Plot
              data={[{
                type: 'bar',
                x: result.per_temperature.map((p) => p.Temperature),
                y: result.per_temperature.map((p) => p.R2),
                text: result.per_temperature.map((p) => p.R2.toFixed(3)),
                marker: { color: '#1b6cf5' },
              }]}
              layout={{
                ...baseLayout, title: t.task3.perTempR2,
                xaxis: { title: t.home.tempAxis },
                shapes: [{
                  type: 'line', xref: 'paper', x0: 0, x1: 1,
                  y0: result.metrics_per_model[result.best_model_name].R2,
                  y1: result.metrics_per_model[result.best_model_name].R2,
                  line: { color: 'red', dash: 'dash' },
                }],
                height: 360,
              }}
              config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
            />
            <Plot
              data={[{
                type: 'bar',
                x: result.per_temperature.map((p) => p.Temperature),
                y: result.per_temperature.map((p) => p.MAE),
                text: result.per_temperature.map((p) => p.MAE.toFixed(3)),
                marker: { color: '#0ea5e9' },
              }]}
              layout={{ ...baseLayout, title: t.task3.perTempMAE, xaxis: { title: t.home.tempAxis }, height: 360 }}
              config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
            />
          </div>

          <hr />
          <h3>{t.task3.gridTitle}</h3>
          <div className="card-caption">
            {t.task3.gridCaption} {result.excluded_aging} · {t.task3.bestModelLabel}: {result.best_model_name}
          </div>
          <NyquistGrid result={result} t={t} />

          <hr />
          <h3>{t.task3.errorsTitle}</h3>
          <div className="cols-2">
            <Plot
              data={[{ type: 'histogram', x: result.error_distribution, nbinsx: 50, marker: { color: '#1b6cf5' } }]}
              layout={{
                ...baseLayout, title: t.task3.errDistTitle,
                shapes: [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: result.error_p95, x1: result.error_p95, line: { color: 'red', dash: 'dash' } }],
                annotations: [{ text: `p95 = ${result.error_p95.toFixed(3)}`, x: result.error_p95, y: 1, yref: 'paper', showarrow: false, xanchor: 'left', font: { color: 'red' } }],
                height: 360,
              }}
              config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
            />
            <Plot
              data={(() => {
                const bySoc = new Map<number, { x: number[]; y: number[] }>()
                result.error_vs_freq.forEach((p) => {
                  const arr = bySoc.get(p.SOC) ?? { x: [], y: [] }
                  arr.x.push(p.Frequency); arr.y.push(p.err)
                  bySoc.set(p.SOC, arr)
                })
                return [...bySoc.entries()].map(([soc, { x, y }]) => ({
                  type: 'scatter' as const, mode: 'markers' as const, x, y,
                  name: `SOC ${soc}`, marker: { size: 4 },
                }))
              })()}
              layout={{
                ...baseLayout, title: t.task3.errFreqTitle,
                xaxis: { title: t.task3.freqAxis, type: 'log' },
                yaxis: { title: '|err| [mΩ]' },
                height: 360,
              }}
              config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
            />
          </div>
        </>
      )}
    </>
  )
}

function NyquistGrid({ result, t }: { result: Task3Response; t: ReturnType<typeof useT> }) {
  const baseLayout = usePlotLayout()
  const temps = result.temperatures
  const socs = result.socs
  const rows = temps.length
  const cols = socs.length
  const traces: any[] = []
  const annotations: any[] = []
  const layoutAxes: Record<string, any> = {}

  result.grid.forEach((row, ri) => {
    row.cells.forEach((cell, ci) => {
      const idx = ri * cols + ci + 1
      const xKey = `xaxis${idx === 1 ? '' : idx}`
      const yKey = `yaxis${idx === 1 ? '' : idx}`
      const xRef = idx === 1 ? 'x' : `x${idx}`
      const yRef = idx === 1 ? 'y' : `y${idx}`

      annotations.push({
        text: `T=${row.temperature}°C · SOC=${socs[ci]}`, showarrow: false,
        x: 0.5, y: 1.08, xref: `${xRef} domain`, yref: `${yRef} domain`,
        font: { size: 9 },
      })
      layoutAxes[xKey] = { showticklabels: true, tickfont: { size: 7 } }
      layoutAxes[yKey] = { showticklabels: true, tickfont: { size: 7 } }
      if (!cell) return

      traces.push({
        type: 'scatter', mode: 'lines+markers',
        x: cell.z_real_actual, y: cell.z_imag_neg_actual,
        xaxis: xRef, yaxis: yRef,
        line: { color: cell.color, width: 1.5 }, marker: { size: 3 },
        showlegend: false, name: t.common.real,
      })
      traces.push({
        type: 'scatter', mode: 'lines+markers',
        x: cell.z_real_pred, y: cell.z_imag_neg_pred,
        xaxis: xRef, yaxis: yRef,
        line: { color: '#888', dash: 'dash', width: 1.2 },
        marker: { size: 3, symbol: 'x' },
        showlegend: false, name: t.common.predicted,
      })
    })
  })

  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout, grid: { rows, columns: cols, pattern: 'independent' },
        ...layoutAxes, annotations,
        height: 180 * rows, margin: { l: 30, r: 10, t: 60, b: 30 },
      }}
      config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
    />
  )
}
