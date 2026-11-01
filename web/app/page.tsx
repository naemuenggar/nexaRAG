"use client";

import { FormEvent, Fragment, ReactNode, useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  ApiError,
  Citation,
  Conversation,
  DocumentItem,
  getToken,
  Message,
  setToken,
  User,
} from "@/lib/api";

function errorText(error: unknown): string {
  return error instanceof ApiError || error instanceof Error ? error.message : "Something went wrong";
}

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    if (!getToken()) {
      setChecking(false);
      return;
    }
    api
      .me()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setChecking(false));
  }, []);

  if (checking) return null;
  if (!user) return <Auth onSignedIn={setUser} />;
  return (
    <Workspace
      user={user}
      onSignOut={() => {
        setToken(null);
        setUser(null);
      }}
    />
  );
}

function Auth({ onSignedIn }: { onSignedIn: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") await api.register(email, password);
      await api.login(email, password);
      onSignedIn(await api.me());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth">
      <h1>nexaRAG</h1>
      <p className="muted">Ask questions about your PDFs. Every answer cites its source page.</p>
      <form onSubmit={submit}>
        <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
        <input
          type="password"
          placeholder={mode === "register" ? "Password (12+ characters)" : "Password"}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          minLength={mode === "register" ? 12 : undefined}
          required
          autoComplete={mode === "register" ? "new-password" : "current-password"}
        />
        {error && <div className="error">{error}</div>}
        <button className="primary" disabled={busy}>
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </button>
      </form>
      <p className="small muted">
        {mode === "login" ? "No account yet? " : "Already registered? "}
        <button className="link" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login" ? "Create one" : "Sign in"}
        </button>
      </p>
    </main>
  );
}

function Workspace({ user, onSignOut }: { user: User; onSignOut: () => void }) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const loadDocuments = useCallback(() => api.documents().then(setDocuments).catch((e) => setError(errorText(e))), []);
  const loadConversations = useCallback(() => api.conversations().then(setConversations).catch(() => undefined), []);

  useEffect(() => {
    loadDocuments();
    loadConversations();
  }, [loadDocuments, loadConversations]);

  // Poll while any document is still being ingested.
  useEffect(() => {
    if (!documents.some((d) => d.status === "pending" || d.status === "processing")) return;
    const timer = setTimeout(loadDocuments, 3000);
    return () => clearTimeout(timer);
  }, [documents, loadDocuments]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  async function openConversation(id: string) {
    setActive(id);
    setError(null);
    try {
      setMessages((await api.conversation(id)).messages);
    } catch (e) {
      setError(errorText(e));
    }
  }

  async function upload(files: FileList | null) {
    if (!files) return;
    setError(null);
    for (const file of Array.from(files)) {
      try {
        await api.upload(file);
      } catch (e) {
        setError(`${file.name}: ${errorText(e)}`);
      }
    }
    if (fileInput.current) fileInput.current.value = "";
    loadDocuments();
  }

  async function remove(doc: DocumentItem) {
    if (!confirm(`Delete ${doc.filename}?`)) return;
    try {
      await api.deleteDocument(doc.id);
      loadDocuments();
    } catch (e) {
      setError(errorText(e));
    }
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || asking) return;
    setAsking(true);
    setError(null);
    setQuestion("");
    let conversationId = active;
    try {
      if (!conversationId) {
        conversationId = (await api.createConversation()).id;
        setActive(conversationId);
        setMessages([]);
      }
      setMessages((m) => [...m, { id: "pending", role: "user", content: text, answered: null, refusal_reason: null, citations: null, created_at: "" }]);
      const result = await api.send(conversationId, text);
      setMessages((m) => [...m.filter((x) => x.id !== "pending"), result.question, result.answer]);
      loadConversations();
    } catch (e) {
      setMessages((m) => m.filter((x) => x.id !== "pending"));
      setQuestion(text);
      setError(errorText(e));
    } finally {
      setAsking(false);
    }
  }

  const ready = documents.filter((d) => d.status === "ready").length;

  return (
    <div className="app">
      <aside className="sidebar">
        <header>
          <strong>nexaRAG</strong>
          <span className="small muted">
            {user.email} · <button className="link" onClick={onSignOut}>Sign out</button>
          </span>
        </header>
        <div className="section">
          <h2>
            Conversations
            <button className="link small" onClick={() => { setActive(null); setMessages([]); }}>+ New</button>
          </h2>
          <ul className="list">
            {conversations.length === 0 && <li className="small muted">None yet</li>}
            {conversations.map((c) => (
              <li key={c.id} className={`clickable ${c.id === active ? "active" : ""}`} onClick={() => openConversation(c.id)}>
                <span className="name">{c.title}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="section docs">
          <h2>
            Documents
            <button className="link small" onClick={() => fileInput.current?.click()}>+ Upload PDF</button>
          </h2>
          <input ref={fileInput} type="file" accept="application/pdf" multiple hidden onChange={(e) => upload(e.target.files)} />
          <ul className="list">
            {documents.length === 0 && <li className="small muted">Upload a PDF to get started</li>}
            {documents.map((d) => (
              <li key={d.id} title={d.error_message ?? undefined}>
                <span className="name">{d.filename}</span>
                <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span className={`badge ${d.status}`}>{d.access === "shared" ? "shared · " : ""}{d.status}</span>
                  {d.access === "owner" && <button className="link small" onClick={() => remove(d)} aria-label={`Delete ${d.filename}`}>✕</button>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && !asking && (
            <div className="empty">
              <h2>Ask your documents</h2>
              <p className="muted">
                {ready > 0
                  ? `${ready} document${ready === 1 ? "" : "s"} ready. Answers cite the file and page they come from, or say they don't know.`
                  : "Upload a PDF on the left. Once it shows “ready”, ask anything about it."}
              </p>
            </div>
          )}
          {messages.map((m) => <Bubble key={m.id} message={m} />)}
          {asking && <div className="bubble muted">Searching your documents…</div>}
          <div ref={bottom} />
        </div>
        {error && <div className="error" style={{ padding: "0 24px 8px" }}>{error}</div>}
        <form className="composer" onSubmit={ask}>
          <textarea
            value={question}
            placeholder="Ask a question about your documents…"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) ask(e);
            }}
            maxLength={2000}
          />
          <button className="primary" disabled={asking || !question.trim()}>Ask</button>
        </form>
      </main>
    </div>
  );
}

function withCitations(text: string): ReactNode[] {
  return text.split(/(\[\d{1,3}\])/).map((part, i) =>
    /^\[\d{1,3}\]$/.test(part) ? <span key={i} className="cite">{part.slice(1, -1)}</span> : <Fragment key={i}>{part}</Fragment>,
  );
}

function pages(c: Citation): string {
  return c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}-${c.page_end}`;
}

function Bubble({ message }: { message: Message }) {
  if (message.role === "user") return <div className="bubble user">{message.content}</div>;
  const refused = message.answered === false;
  return (
    <div className={`bubble ${refused ? "refused" : ""}`}>
      {withCitations(message.content)}
      {!!message.citations?.length && (
        <div className="sources">
          {message.citations.map((c) => (
            <div key={c.number} className="source">
              <span className="cite">{c.number}</span> <strong>{c.filename}</strong>{" "}
              <span className="muted">{pages(c)}</span>
              <div className="muted">“{c.snippet}”</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
