import { useEffect, useMemo, useState } from 'react'
import { fetchOptions, runTask1 } from '../api'
import type { DatasetOptions, Task1Response } from '../types'
import Plot, { baseConfig, usePlotLayout } from '../components/Plot'
import MetricGrid from '../components/MetricGrid'
import MetricBar from '../components/MetricBar'
import MetricsTable from '../components/MetricsTable'
import Segmented from '../components/Segmented'
import Spinner from '../components/Spinner'
import { useT } from '../i18n'

type Mode = 'manual' | 'random'

export default function Task1() {
  const t = useT()
  const baseLayout = usePlotLayout()
  const [opts, setOpts] = useState<DatasetOptions | null>(null)
  const [mode, setMode] = useState<Mode>('manual')
  const [aging, setAging] = useState<number>(2)
  const [temp, setTemp] = useState<number>(22.5)
  const [soc, setSoc] = useState<number>(2)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Task1Response | null>(null)
  const [focusSoc, setFocusSoc] = useState<number>(2)
  const [showAll, setShowAll] = useState(false)
  const [innerTab, setInnerTab] = useState<'real' | 'pred' | 'overlay'>('real')
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState<number | null>(null)

  useEffect(() => { fetchOptions().then(setOpts) }, [])

  function reroll() {
    if (!opts) return
    const pick = <T,>(arr: T[]) => arr[Math.floor(Math.random() * arr.length)]
    setAging(pick(opts.agings))
    setTemp(pick(opts.temperatures))
    setSoc(pick(opts.socs))
  }

  useEffect(() => {
    if (mode === 'random' && opts) reroll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, opts])

  async function run() {
    setRunning(true); setError(null); setResult(null)
    const t0 = performance.now()
    try {
      const data = await runTask1(aging, temp)
      setResult(data); setFocusSoc(soc)
      setElapsed((performance.now() - t0) / 1000)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || t.common.runFailed)
    } finally { setRunning(false) }
  }

  const focusKeys = useMemo(() => {
    if (!result) return []
    return showAll ? result.available_socs.map(String) : [String(focusSoc)]
  }, [result, focusSoc, showAll])

  const modeOptions = [t.common.manual, t.common.random] as const

  return (
    <>
      <h1 className="page-title">{t.task1.title}</h1>
      <div className="page-caption">{t.task1.caption}</div>
      <div className="banner">{t.task1.banner}</div>

      <div className="card">
        <div className="card-title">{t.common.configuration}</div>
        <div className="card-caption">{t.task1.cardCaption}</div>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <Segmented
            value={mode === 'manual' ? t.common.manual : t.common.random}
            options={modeOptions}
            onChange={(v) => setMode(v === t.common.manual ? 'manual' : 'random')}
          />
        </div>

        {mode === 'manual' ? (
          <div className="row">
            <label className="field">{t.task1.aging}
              <select value={aging} onChange={(e) => setAging(Number(e.target.value))}>
                {opts?.agings.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </label>
            <label className="field">{t.task1.temperature}
              <select value={temp} onChange={(e) => setTemp(Number(e.target.value))}>
                {opts?.temperatures.map((tt) => <option key={tt} value={tt}>{tt}</option>)}
              </select>
            </label>
            <label className="field">{t.task1.soc}
              <select value={soc} onChange={(e) => setSoc(Number(e.target.value))}>
                {opts?.socs.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
          </div>
        ) : (
          <div className="row">
            <div className="pick-row">
              <span className="pick-chip">{t.task1.aging} <b>{aging}</b></span>
              <span className="pick-chip">{t.task1.temperature.replace(' [°C]','')} <b>{temp}°C</b></span>
              <span className="pick-chip">{t.task1.soc} <b>{soc}</b></span>
            </div>
            <button onClick={reroll} type="button">{t.common.reroll}</button>
          </div>
        )}

        <div className="row end" style={{ marginTop: '0.9rem' }}>
          <button className="primary" onClick={run} disabled={running}>
            {running ? <Spinner label={t.task1.spinnerLabel} /> : t.task1.runBtn}
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
          <h3>{t.task1.explainHeader}</h3>
          <ol>{t.task1.explain.map((s, i) => <li key={i}>{s}</li>)}</ol>
        </div>
      )}

      {result && (
        <>
          <h3 style={{ marginTop: '1.5rem' }}>{t.task1.bestPerf}</h3>
          <div className="card-caption">
            {t.task1.excluded}: Aging {result.excluded_aging} · {result.excluded_temperature}°C
          </div>
          <MetricGrid
            items={[
              [t.common.bestModel, result.best_model_name],
              [t.task1.r2mean, result.metrics_per_model[result.best_model_name].R2],
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
              sortKey="MSE" ascending highlight={result.best_model_name}
            />
          </div>

          <hr />
          <h3>{t.task1.reconstructTitle}</h3>
          <div className="card-caption">
            {t.task1.reconstructCaption.replace('{a}', String(result.excluded_aging)).replace('{t}', String(result.excluded_temperature))}
          </div>

          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <label className="field">{t.task1.socFocus}
              <select value={focusSoc} onChange={(e) => setFocusSoc(Number(e.target.value))} disabled={showAll}>
                {result.available_socs.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
            <label className="field"><span>{t.task1.mode}</span>
              <button onClick={() => setShowAll((v) => !v)}>
                {showAll ? t.task1.showAll : t.task1.showOne}
              </button>
            </label>
          </div>

          <div className="tabs">
            <button className={innerTab === 'real' ? 'active' : ''} onClick={() => setInnerTab('real')}>{t.common.real}</button>
            <button className={innerTab === 'pred' ? 'active' : ''} onClick={() => setInnerTab('pred')}>{t.common.predicted}</button>
            <button className={innerTab === 'overlay' ? 'active' : ''} onClick={() => setInnerTab('overlay')}>{t.common.overlay}</button>
          </div>

          <div className="card">
            <Plot
              data={focusKeys.flatMap<any>((k) => {
                const s = result.predictions_by_soc[k]
                if (!s) return []
                if (innerTab === 'real') {
                  return [{ type: 'scatter', mode: 'lines+markers', x: s.z_real_actual, y: s.z_imag_neg_actual, name: `SOC ${s.soc}`, hovertemplate: `SOC=${s.soc}<br>Re=%{x:.3f}<br>-Im=%{y:.3f}<extra></extra>` }]
                }
                if (innerTab === 'pred') {
                  return [{ type: 'scatter', mode: 'lines+markers', x: s.z_real_pred, y: s.z_imag_neg_pred, name: `SOC ${s.soc} — ${t.common.predicted}`, line: { dash: 'dash' }, marker: { symbol: 'x' } }]
                }
                return [
                  { type: 'scatter', mode: 'lines+markers', x: s.z_real_actual, y: s.z_imag_neg_actual, name: `SOC ${s.soc} — ${t.common.real}`, legendgroup: `soc-${s.soc}` },
                  { type: 'scatter', mode: 'lines+markers', x: s.z_real_pred, y: s.z_imag_neg_pred, name: `SOC ${s.soc} — ${t.common.predicted}`, legendgroup: `soc-${s.soc}`, line: { dash: 'dash' }, marker: { symbol: 'x' } },
                ]
              })}
              layout={{
                ...baseLayout,
                title: innerTab === 'real'
                  ? `${t.common.real} — Aging ${result.excluded_aging} · ${result.excluded_temperature}°C`
                  : innerTab === 'pred'
                  ? `${t.common.predicted}: ${result.best_model_name}`
                  : `${result.best_model_name} · ${t.common.overlay}`,
                xaxis: { title: 'Re(Z) [mΩ]' },
                yaxis: { title: '-Im(Z) [mΩ]', scaleanchor: 'x', scaleratio: 1 },
                height: 500,
                legend: { orientation: 'h', y: -0.2 },
              }}
              config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
            />
          </div>

          <hr />
          <h3>{t.task1.residTitle}</h3>
          <table>
            <thead>
              <tr><th>SOC</th><th>MAE Re(Z)</th><th>MAE Im(Z)</th><th>Max |err|</th><th>{t.task1.points}</th></tr>
            </thead>
            <tbody>
              {result.residuals_by_soc.map((r) => (
                <tr key={r.soc}>
                  <td>{r.soc}</td><td>{r.mae_real.toFixed(4)}</td><td>{r.mae_imag.toFixed(4)}</td>
                  <td>{r.max_abs.toFixed(4)}</td><td>{r.n_points}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  )
}
