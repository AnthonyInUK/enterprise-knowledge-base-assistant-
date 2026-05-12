import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Braces,
  Clock,
  Database,
  FileText,
  GitBranch,
  Loader2,
  Search,
  Send,
  ShieldCheck
} from "lucide-react";
import "./styles.css";

type RetrievedChunk = {
  rank: number;
  score: number;
  chunk_id: string;
  document_title: string;
  chunk_level: string;
  page_start: number | null;
  page_end: number | null;
  section_title: string | null;
  text: string;
  matched_terms: string[];
  reason: string;
};

type ResearchAnswer = {
  question: string;
  answer: string;
  sources: string[];
  retrieved_chunks: RetrievedChunk[];
  latency_ms: number;
  used_llm: boolean;
  debug: Record<string, unknown>;
  facts: Array<Record<string, unknown>>;
};

type ReviewItem = {
  id: string;
  status: string;
  reviewed_fact: string;
  corrected_fact: string | null;
  source_citation: string | null;
  metadata: Record<string, unknown>;
};

type OpsMetrics = {
  request_count_24h: number;
  error_count_24h: number;
  avg_latency_ms_24h: number | null;
  p95_latency_ms_24h: number | null;
  query_count_24h: number;
  avg_query_latency_ms_24h: number | null;
  pending_reviews: number;
  recent_errors: Array<Record<string, unknown>>;
};

const examples = [
  "What was Tesla's automotive revenue in 2023?",
  "How many vehicles did Tesla deliver in 2023?",
  "What does Tesla say about its Supercharger network?",
  "宁德时代2023年营业收入是多少？",
  "Compare Tesla and CATL on 2023 revenue and business highlights."
];

function formatMs(value?: number | null) {
  if (value === null || value === undefined) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

function evidenceExcerpt(chunk: RetrievedChunk) {
  const text = chunk.text.replace(/\s+/g, " ").trim();
  const terms = chunk.matched_terms
    .filter((term) => term.length > 2)
    .sort((a, b) => b.length - a.length);
  const firstHit = terms
    .map((term) => text.toLowerCase().indexOf(term.toLowerCase()))
    .filter((index) => index >= 0)
    .sort((a, b) => a - b)[0];
  const start = firstHit === undefined ? 0 : Math.max(0, firstHit - 90);
  const excerpt = text.slice(start, start + 760);
  return `${start > 0 ? "... " : ""}${excerpt}${start + 760 < text.length ? " ..." : ""}`;
}

function App() {
  const [question, setQuestion] = useState(examples[0]);
  const [topK, setTopK] = useState(5);
  const [apiKey, setApiKey] = useState("");
  const [result, setResult] = useState<ResearchAnswer | null>(null);
  const [metrics, setMetrics] = useState<OpsMetrics | null>(null);
  const [selectedRank, setSelectedRank] = useState(1);
  const [reviews, setReviews] = useState<ReviewItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedChunk = useMemo(
    () => result?.retrieved_chunks.find((chunk) => chunk.rank === selectedRank) ?? result?.retrieved_chunks[0],
    [result, selectedRank]
  );

  async function fetchMetrics() {
    try {
      const response = await fetch("/v1/ops/metrics", {
        headers: apiKey ? { "x-api-key": apiKey } : undefined
      });
      if (response.ok) {
        setMetrics(await response.json());
      }
    } catch {
      setMetrics(null);
    }
  }

  async function fetchReviews() {
    try {
      const response = await fetch("/v1/reviews?status=pending&limit=5", {
        headers: apiKey ? { "x-api-key": apiKey } : undefined
      });
      if (response.ok) {
        setReviews(await response.json());
      }
    } catch {
      setReviews([]);
    }
  }

  useEffect(() => {
    fetchMetrics();
    fetchReviews();
    const timer = window.setInterval(() => {
      fetchMetrics();
      fetchReviews();
    }, 20000);
    return () => window.clearInterval(timer);
  }, [apiKey]);

  async function updateReview(reviewId: string, status: "approved" | "rejected" | "needs_revision") {
    const response = await fetch(`/v1/reviews/${reviewId}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "x-api-key": apiKey } : {})
      },
      body: JSON.stringify({ status, reviewer_id: "analyst-ui" })
    });
    if (response.ok) {
      fetchReviews();
      fetchMetrics();
    }
  }

  async function ask() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/v1/research/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { "x-api-key": apiKey } : {})
        },
        body: JSON.stringify({
          question,
          top_k: topK,
          thread_id: "research-ui",
          user_id: "analyst-ui"
        })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Request failed (${response.status})`);
      }
      const payload = (await response.json()) as ResearchAnswer;
      setResult(payload);
      setSelectedRank(payload.retrieved_chunks[0]?.rank ?? 1);
      fetchMetrics();
      fetchReviews();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="appleShell">
      <aside className="sidebar">
        <div className="windowDots" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div className="brandBlock">
          <p className="eyebrow">Renewable Energy</p>
          <h1>Research Agent</h1>
        </div>

        <section className="queryPanel">
          <div className="panelHeader">
            <Search size={17} />
            <span>Ask</span>
          </div>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={4}
            placeholder="Ask about revenue, shipments, technology, markets, or risks"
          />
          <div className="compactControls">
            <label>
              Top-K
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>
            <label className="apiField">
              API key
              <input
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder="optional"
              />
            </label>
          </div>
          <button className="primary" onClick={ask} disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            Ask with evidence
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="examples">
          <p>Examples</p>
          {examples.map((item) => (
            <button key={item} onClick={() => setQuestion(item)}>
              {item}
            </button>
          ))}
        </section>

        <section className="sidebarMetrics">
          <Metric icon={<Database />} label="Requests" value={metrics?.request_count_24h ?? 0} />
          <Metric icon={<AlertTriangle />} label="Errors" value={metrics?.error_count_24h ?? 0} />
          <Metric icon={<Clock />} label="P95" value={formatMs(metrics?.p95_latency_ms_24h)} />
          <Metric icon={<ShieldCheck />} label="Reviews" value={metrics?.pending_reviews ?? 0} />
        </section>
      </aside>

      <section className="mainStage">
        <header className="heroBar">
          <div>
            <p className="eyebrow">Evidence-grounded industry research</p>
            <h2>Answers with sources, facts, and traceable evidence.</h2>
          </div>
          <div className="heroPills">
            <span>Hybrid+Rerank</span>
            <span>Hit@5 100%</span>
            <span>MRR 0.915</span>
          </div>
        </header>

        <section className="contentGrid">
          <article className="answerPanel">
            <div className="panelHeader">
              <BookOpen size={17} />
              <span>Answer</span>
            </div>
          {result ? (
            <>
              <div className="answerMeta">
                <span>{formatMs(result.latency_ms)}</span>
                <span>{result.used_llm ? "LLM generated" : "Direct extractive"}</span>
                <span>{String(result.debug.rerank_backend ?? "rerank")}</span>
              </div>
              <article className="answerText">{result.answer}</article>
              <div className="sources">
                <h2>Sources</h2>
                {result.sources.map((source) => (
                  <div className="source" key={source}>{source}</div>
                ))}
              </div>
              {result.facts.length > 0 && (
                <div className="facts">
                  <h2>Structured facts</h2>
                  {result.facts.map((fact) => (
                    <div className="factRow" key={String(fact.id)}>
                      <GitBranch size={15} />
                      <span>{String(fact.company)} · {String(fact.metric)} · {String(fact.value)} {String(fact.unit ?? "")}</span>
                      <strong>{String(fact.review_status)}</strong>
                    </div>
                  ))}
                </div>
              )}
              {reviews.length > 0 && (
                <div className="reviewBox">
                  <h2>Pending review</h2>
                  {reviews.map((review) => (
                    <div className="reviewItem" key={review.id}>
                      <p>{review.reviewed_fact}</p>
                      {review.source_citation && <span>{review.source_citation}</span>}
                      <div>
                        <button onClick={() => updateReview(review.id, "approved")}>Approve</button>
                        <button onClick={() => updateReview(review.id, "needs_revision")}>Revise</button>
                        <button onClick={() => updateReview(review.id, "rejected")}>Reject</button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="empty">
              <FileText size={36} />
              <p>Ask a question to generate a cited answer.</p>
            </div>
          )}
          </article>

          <aside className="evidencePanel">
            <div className="panelHeader">
              <Braces size={17} />
              <span>Evidence</span>
            </div>
          {result ? (
            <>
              <div className="rankTabs">
                {result.retrieved_chunks.map((chunk) => (
                  <button
                    key={chunk.chunk_id}
                    className={chunk.rank === selectedRank ? "active" : ""}
                    onClick={() => setSelectedRank(chunk.rank)}
                  >
                    {chunk.rank}
                  </button>
                ))}
              </div>
              {selectedChunk && (
                <div className="chunkCard">
                  <div className="chunkTitle">
                    <strong>{selectedChunk.document_title}</strong>
                    <span>score {selectedChunk.score.toFixed(3)}</span>
                  </div>
                  <p className="pageLine">
                    {selectedChunk.section_title || "Overview"}
                    {selectedChunk.page_start ? ` · p.${selectedChunk.page_start}` : ""}
                  </p>
                  <p className="chunkText">{evidenceExcerpt(selectedChunk)}</p>
                  <div className="terms">
                    {selectedChunk.matched_terms.slice(0, 10).map((term) => (
                      <span key={term}>{term}</span>
                    ))}
                  </div>
                </div>
              )}
              <div className="debugBox">
                <div className="debugTitle">
                  <Activity size={16} />
                  Debug
                </div>
                <pre>{JSON.stringify(result.debug, null, 2)}</pre>
              </div>
            </>
          ) : (
            <div className="empty small">
              <BarChart3 size={30} />
              <p>Evidence chunks and debug metadata will appear here.</p>
            </div>
          )}
          </aside>
        </section>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="metric">
      {icon}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
