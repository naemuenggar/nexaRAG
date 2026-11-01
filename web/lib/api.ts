// Thin client for the nexaRAG API. The browser talks to the API directly (CORS), so large
// PDF uploads never pass through a Vercel function and its request-size limit.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const TOKEN_KEY = "nexarag.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private mode: the session simply won't survive a reload.
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the API. Is the backend running?");
  }
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body?.error;
    if (response.status === 401) setToken(null);
    throw new ApiError(response.status, error?.code ?? "error", error?.message ?? `HTTP ${response.status}`);
  }
  return body as T;
}

export type User = { id: string; email: string };

export type DocumentItem = {
  id: string;
  filename: string;
  size_bytes: number;
  page_count: number | null;
  status: "pending" | "processing" | "ready" | "failed";
  error_message: string | null;
  access: "owner" | "shared";
  created_at: string;
};

export type Citation = {
  number: number;
  document_id: string;
  filename: string;
  page_start: number;
  page_end: number;
  snippet: string;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  answered: boolean | null;
  refusal_reason: string | null;
  citations: Citation[] | null;
  created_at: string;
};

export type Conversation = { id: string; title: string; updated_at: string };

export const api = {
  register: (email: string, password: string) =>
    request<User>("/auth/register", { method: "POST", body: JSON.stringify({ email, password }) }),
  login: async (email: string, password: string) => {
    const result = await request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setToken(result.access_token);
  },
  me: () => request<User>("/auth/me"),
  documents: () => request<{ items: DocumentItem[] }>("/documents?limit=100").then((r) => r.items),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentItem>("/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  conversations: () =>
    request<{ items: Conversation[] }>("/conversations?limit=50").then((r) => r.items),
  createConversation: () => request<Conversation>("/conversations", { method: "POST", body: "{}" }),
  conversation: (id: string) =>
    request<Conversation & { messages: Message[] }>(`/conversations/${id}`),
  send: (id: string, question: string) =>
    request<{ question: Message; answer: Message; latency: { total_ms: number } }>(
      `/conversations/${id}/messages`,
      { method: "POST", body: JSON.stringify({ question }) },
    ),
};
