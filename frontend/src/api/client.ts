/**
 * Single fetch wrapper: bearer token, JSON handling, uniform error shape,
 * and abort support so a long AI call can be cancelled from the UI.
 */

const TOKEN_KEY = 'mlea.token'

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage disabled (private mode) - the app still works for this session */
  }
}

function qs(params?: Record<string, unknown>): string {
  if (!params) return ''
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    search.append(key, String(value))
  })
  const out = search.toString()
  return out ? `?${out}` : ''
}

interface RequestOptions {
  method?: string
  body?: unknown
  params?: Record<string, unknown>
  signal?: AbortSignal
  formData?: FormData
}

export async function request<T = unknown>(path: string, options: RequestOptions = {}): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`

  let body: BodyInit | undefined
  if (options.formData) {
    body = options.formData
  } else if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(options.body)
  }

  let response: Response
  try {
    response = await fetch(`/api${path}${qs(options.params)}`, {
      method: options.method ?? (options.body || options.formData ? 'POST' : 'GET'),
      headers,
      body,
      signal: options.signal,
    })
  } catch (error) {
    if ((error as Error).name === 'AbortError') throw error
    throw new ApiError(0, 'Backend unreachable. Is the API running on this host?', error)
  }

  if (response.status === 401) {
    throw new ApiError(401, 'Create a profile to start (Setup screen).')
  }

  const text = await response.text()
  let payload: unknown = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }
  if (!response.ok) {
    const detail = (payload as { detail?: unknown })?.detail ?? payload
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? (detail as Array<{ msg?: string }>).map((d) => d.msg ?? JSON.stringify(d)).join('; ')
          : `Request failed with ${response.status}`
    throw new ApiError(response.status, message, payload)
  }
  return payload as T
}

export const api = {
  get: <T,>(path: string, params?: Record<string, unknown>, signal?: AbortSignal) =>
    request<T>(path, { method: 'GET', params, signal }),
  post: <T,>(path: string, body?: unknown, params?: Record<string, unknown>, signal?: AbortSignal) =>
    request<T>(path, { method: 'POST', body, params, signal }),
  patch: <T,>(path: string, body?: unknown) => request<T>(path, { method: 'PATCH', body }),
  del: <T,>(path: string) => request<T>(path, { method: 'DELETE' }),
  upload: <T,>(path: string, formData: FormData, params?: Record<string, unknown>) =>
    request<T>(path, { method: 'POST', formData, params }),
}
