const BASE = "/api";

export function getToken(): string {
  return localStorage.getItem("ku_token") || "";
}

export function setToken(token: string) {
  localStorage.setItem("ku_token", token);
}

export function clearToken() {
  localStorage.removeItem("ku_token");
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...options, headers });
  } catch {
    throw new ApiError(0, "Не удалось соединиться с сервером. Проверь подключение.");
  }
  if (!res.ok) {
    let message = "Не удалось загрузить данные. Попробуем ещё раз.";
    try {
      const data = await res.json();
      if (typeof data.detail === "string") message = data.detail;
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, message);
  }
  return res.json();
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
};
