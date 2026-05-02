import axios from 'axios'
import type {
  AggByTempResponse,
  AgingEvolutionResponse,
  BenchmarkResponse,
  CurvesResponse,
  DatasetOptions,
  DatasetSummary,
  ProjectInfo,
  Task1Response,
  Task2Response,
  Task3Response,
} from './types'

const client = axios.create({
  baseURL: '/api',
  timeout: 600_000,
})

export async function fetchProject(): Promise<ProjectInfo> {
  const { data } = await client.get<ProjectInfo>('/project')
  return data
}

export async function fetchSummary(): Promise<DatasetSummary> {
  const { data } = await client.get<DatasetSummary>('/dataset/summary')
  return data
}

export async function fetchOptions(): Promise<DatasetOptions> {
  const { data } = await client.get<DatasetOptions>('/dataset/options')
  return data
}

export async function fetchCurves(aging: number): Promise<CurvesResponse> {
  const { data } = await client.get<CurvesResponse>('/dataset/curves', {
    params: { aging },
  })
  return data
}

export async function fetchAggByTemp(): Promise<AggByTempResponse> {
  const { data } = await client.get<AggByTempResponse>('/dataset/agg-by-temp')
  return data
}

export async function fetchAgingEvolution(
  soc: number,
  excludedAging: number,
): Promise<AgingEvolutionResponse> {
  const { data } = await client.get<AgingEvolutionResponse>(
    '/dataset/aging-evolution',
    { params: { soc, excluded_aging: excludedAging } },
  )
  return data
}

export async function runTask1(
  aging: number,
  temperature: number,
): Promise<Task1Response> {
  const { data } = await client.post<Task1Response>('/task1/run', {
    aging,
    temperature,
  })
  return data
}

export async function runTask2(
  aging: number,
  soc: number,
): Promise<Task2Response> {
  const { data } = await client.post<Task2Response>('/task2/run', { aging, soc })
  return data
}

export async function runTask3(excludedAging: number): Promise<Task3Response> {
  const { data } = await client.post<Task3Response>('/task3/run', {
    excluded_aging: excludedAging,
  })
  return data
}

export async function fetchBenchmark(
  task: 'task1' | 'task2' | 'task3',
): Promise<BenchmarkResponse> {
  const { data } = await client.get<BenchmarkResponse>(`/benchmarks/${task}`)
  return data
}

export const paperUrl = '/api/paper'
