export interface ProjectInfo {
  name: string
  version: string
  description: string
  authors: string[]
  paper: {
    title: string
    venue: string
    doi: string
    url: string
    available: boolean
  }
  constants: {
    aging_labels: Record<string, string>
    aging_colors: Record<string, string>
    soc_labels: Record<string, string>
    temp_colors: Record<string, string>
    class_names: string[]
    class_colors: string[]
  }
}

export interface DatasetSummary {
  rows: number
  columns: string[]
  agings: number[]
  temperatures: number[]
  socs: number[]
  freq_min: number
  freq_max: number
  n_combinations: number
  n_curves: number
}

export interface DatasetOptions {
  agings: number[]
  temperatures: number[]
  socs: number[]
}

export interface CurveSeries {
  temperature: number
  color: string
  z_real: number[]
  z_imag_neg: number[]
  soc: number[]
  frequency: number[]
}
export interface CurvesResponse {
  aging: number
  series: CurveSeries[]
}

export interface AggByTempSeries {
  aging: number
  label: string
  color: string
  temperature: number[]
  z_real_mean: number[]
}
export interface AggByTempResponse {
  series: AggByTempSeries[]
}

export interface AgingEvolutionTrace {
  aging: number
  label: string
  color: string
  dashed: boolean
  z_real: number[]
  z_imag_neg: number[]
}
export interface AgingEvolutionPanel {
  temperature: number
  traces: AgingEvolutionTrace[]
}
export interface AgingEvolutionResponse {
  soc: number
  excluded_aging: number
  panels: AgingEvolutionPanel[]
}

export interface ModelMetrics {
  [key: string]: number
}

export interface Task1PredictionsBySoc {
  [soc: string]: {
    soc: number
    frequency: number[]
    z_real_actual: number[]
    z_imag_neg_actual: number[]
    z_real_pred: number[]
    z_imag_neg_pred: number[]
  }
}

export interface Task1Response {
  task: 'task1'
  excluded_aging: number
  excluded_temperature: number
  best_model_name: string
  metrics_per_model: Record<string, ModelMetrics>
  training_time_s: Record<string, number>
  n_test_points: number
  available_socs: number[]
  focus_soc: number
  predictions_by_soc: Task1PredictionsBySoc
  residuals_by_soc: {
    soc: number
    mae_real: number
    mae_imag: number
    max_abs: number
    n_points: number
  }[]
}

export interface Task2PanelGroup {
  class: number
  label: string
  color: string
  z_real: number[]
  z_imag_neg: number[]
  frequency: number[]
}
export interface Task2Panel {
  temperature: number
  n_points: number
  n_correct: number
  accuracy: number
  is_correct: boolean
  groups: Task2PanelGroup[]
}
export interface Task2PerTempRow {
  Temperature: number
  N: number
  N_correct: number
  Accuracy: number
  N_predicted_young: number
  N_predicted_old: number
}
export interface Task2Response {
  task: 'task2'
  aging: number
  soc: number
  true_class: number
  true_label: string
  class_names: string[]
  class_colors: string[]
  best_model_name: string
  metrics_per_model: Record<string, ModelMetrics>
  training_time_s: Record<string, number>
  n_test_points: number
  panels: Task2Panel[]
  per_temperature: Task2PerTempRow[]
  feature_importance: Record<string, number> | null
  accuracy_full: number
  errors_full: number
  n_full: number
}

export interface Task3GridCell {
  temperature: number
  soc: number
  color: string
  z_real_actual: number[]
  z_imag_neg_actual: number[]
  z_real_pred: number[]
  z_imag_neg_pred: number[]
}
export interface Task3GridRow {
  temperature: number
  cells: (Task3GridCell | null)[]
}
export interface Task3Response {
  task: 'task3'
  excluded_aging: number
  best_model_name: string
  metrics_per_model: Record<string, ModelMetrics>
  training_time_s: Record<string, number>
  n_test_points: number
  per_temperature: { Temperature: number; R2: number; MAE: number; N: number }[]
  grid: Task3GridRow[]
  error_distribution: number[]
  error_p95: number
  error_vs_freq: { Frequency: number; err: number; SOC: number }[]
  temperatures: number[]
  socs: number[]
}

export type BenchmarkResponse = Record<string, unknown>
