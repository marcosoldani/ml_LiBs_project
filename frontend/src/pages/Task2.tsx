import { useEffect, useState } from 'react'
import { fetchOptions, runTask2 } from '../api'
import type { DatasetOptions, Task2Response } from '../types'
import Plot, { baseConfig, usePlotLayout } from '../components/Plot'
import MetricGrid from '../components/MetricGrid'
import MetricBar from '../components/MetricBar'
import MetricsTable from '../components/MetricsTable'
import Segmented from '../components/Segmented'
import Spinner from '../components/Spinner'
import { useT } from '../i18n'

type Mode = 'manual' | 'random'

export default function Task2() {
  const t = useT()
  const baseLayout = usePlotLayout()
  const [opts, setOpts] = useState<DatasetOptions | null>(null)
  const [mode, setMode] = useState<Mode>('manual')
  const [aging, setAging] = useState<number>(4)
  const [soc, setSoc] = useState<number>(3)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Task2Response | null>(null)
  const [elapsed, setElapsed] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { fetchOptions().then(setOpts) }, [])

  function reroll() {
    if (!opts) return
    setAging(opts.agings[Math.floor(Math.random() * opts.agings.length)])
    setSoc(opts.socs[Math.floor(Math.random() * opts.socs.length)])
  }

  useEffect(() => {
    if (mode === 'random' && opts) reroll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, opts])

  async function run() {
    setRunning(true); setError(null); setResult(null)
    const t0 = performance.now()
    try {
      const data = await runTask2(aging, soc); setResult(data)
      setElapsed((performance.now() - t0) / 1000)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || t.common.runFailed)
    } finally { setRunning(false) }
  }

  const modeOptions = [t.common.manual, t.common.random] as const

  return (
    <>
      <h1 className="page-title">{t.task2.title}</h1>
      <div className="page-caption">{t.task2.caption}</div>
      <div className="banner">{t.task2.banner}</div>

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
          <div className="row">
            <label className="field">{t.task2.aging}
              <select value={aging} onChange={(e) => setAging(Number(e.target.value))}>
                {opts?.agings.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </label>
            <label className="field">{t.task2.socSel}
              <select value={soc} onChange={(e) => setSoc(Number(e.target.value))}>
                {opts?.socs.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
          </div>
        ) : (
          <div className="row">
            <span className="pick-chip">{t.task2.aging} <b>{aging}</b></span>
            <span className="pick-chip">{t.task2.socSel} <b>{soc}</b></span>
            <button onClick={reroll}>{t.common.reroll}</button>
          </div>
        )}
        <div className="row end" style={{ marginTop: '0.9rem' }}>
          <button className="primary" onClick={run} disabled={running}>
            {running ? <Spinner label={t.task2.spinnerLabel} /> : t.task2.runBtn}
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
          <h3>{t.task2.explainHeader}</h3>
          <ol>{t.task2.explain.map((s, i) => <li key={i}>{s}</li>)}</ol>
        </div>
      )}

      {result && (
        <>
          <h3 style={{ marginTop: '1.5rem' }}>{t.task2.bestClassifier}</h3>
          <div className="card-caption">
            Aging = {result.aging} · SOC = {result.soc} · {t.task2.realClass}: <strong>{result.true_label}</strong>
          </div>
          <MetricGrid
            items={[
              [t.common.bestModel, result.best_model_name],
              ['Accuracy', result.metrics_per_model[result.best_model_name].Accuracy],
              [t.task2.accuracyOn, result.accuracy_full],
              [t.common.trainingTime, `${result.training_time_s[result.best_model_name].toFixed(2)}s`],
              [t.common.testPoints, result.n_test_points],
            ]}
          />

          <hr />
          <h3>{t.common.benchmark}</h3>
          <div className="cols-2">
            <MetricBar metrics={result.metrics_per_model} metricKey="Accuracy" title="Accuracy" />
          </div>
          <div style={{ marginTop: '0.8rem' }}>
            <MetricsTable
              metrics={result.metrics_per_model}
              trainingTime={result.training_time_s}
              sortKey="Accuracy"
              highlight={result.best_model_name}
            />
          </div>

          <hr />
          <h3>{t.task2.perTempTitle}</h3>
          <Plot
            data={[{
              type: 'bar',
              x: result.per_temperature.map((p) => `${p.Temperature}°C`),
              y: result.per_temperature.map((p) => p.Accuracy),
              text: result.per_temperature.map((p) => `${p.N_correct}/${p.N}`),
              textposition: 'outside',
              marker: {
                color: result.per_temperature.map((p) =>
                  p.Accuracy >= 0.95 ? '#15803d' : p.Accuracy >= 0.7 ? '#b45309' : '#dc2626',
                ),
              },
            }]}
            layout={{
              ...baseLayout,
              title: t.task2.perTempAcc,
              yaxis: { title: 'Accuracy', range: [0, 1.05] },
              height: 360,
            }}
            config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
          />

          <hr />
          <h3>{t.task2.gridTitle}</h3>
          <div className="card-caption">{t.task2.gridCaption}</div>
          <div className="cols-2">
            {result.panels.map((panel) => {
              const isCorrect = panel.is_correct
              return (
                <div key={panel.temperature} className="card">
                  <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>T = {panel.temperature}°C</span>
                    <span className="pick-chip" style={{
                      background: isCorrect ? 'var(--success-soft)' : 'var(--danger-soft)',
                      borderColor: isCorrect ? 'var(--success)' : 'var(--danger)',
                    }}>
                      {panel.n_correct}/{panel.n_points} · {(panel.accuracy * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Plot
                    data={panel.groups
                      .filter((g) => g.z_real.length > 0)
                      .map((g) => ({
                        type: 'scatter',
                        mode: 'markers',
                        x: g.z_real,
                        y: g.z_imag_neg,
                        name: `${t.common.predicted}: ${g.label}`,
                        marker: { color: g.color, size: 6, opacity: 0.8 },
                      }))}
                    layout={{
                      ...baseLayout,
                      xaxis: { title: 'Re(Z) [mΩ]' },
                      yaxis: { title: '-Im(Z) [mΩ]', scaleanchor: 'x', scaleratio: 1 },
                      height: 280,
                      margin: { l: 50, r: 10, t: 10, b: 40 },
                      legend: { orientation: 'h', y: -0.25 },
                    }}
                    config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
                  />
                </div>
              )
            })}
          </div>
          <div className="banner" style={{ marginTop: '1rem' }}>
            {t.task2.accuracyOn}: <strong>{result.accuracy_full.toFixed(4)}</strong>
            {' '}({result.errors_full} {t.task2.errorsOf} {result.n_full})
          </div>

          {result.feature_importance && (
            <>
              <hr />
              <h3>{t.task2.fiTitle} ({result.best_model_name})</h3>
              <Plot
                data={[{
                  type: 'bar', orientation: 'h',
                  x: Object.values(result.feature_importance),
                  y: Object.keys(result.feature_importance),
                  marker: { color: '#1b6cf5' },
                }]}
                layout={{ ...baseLayout, height: 380, xaxis: { title: 'Importance' }, yaxis: { autorange: 'reversed' } }}
                config={baseConfig as any} useResizeHandler style={{ width: '100%' }}
              />
            </>
          )}
        </>
      )}
    </>
  )
}
