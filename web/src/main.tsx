import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import { SceneViewer } from "./SceneViewer";

type Run = {
  id: string;
  owner: string;
  status: string;
  dataset: string;
  scene_id: string;
  preset: string;
  start_at: string;
  stop_after: string;
  gpu_id?: string;
  error?: string;
  created_at?: string;
  finished_at?: string;
};
type Preset = { slug: string; name: string; datasets: string[] };
type Navigate = (path: string) => void;

const stages = ["prepare", "train", "stereo", "bucket", "fit", "inspect", "uncertainty", "map-viz", "render", "nbv", "nbv-viz", "bundle"];

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error((await response.json().catch(() => ({ detail: response.statusText }))).detail || response.statusText);
  }
  return response.json();
}

function Header({ title, navigate }: { title: string; navigate: Navigate }) {
  const isCatalog = title === "Trained runs";
  return <header>
    <div>
      <p className="eyebrow">VBOGS CONTROL PLANE</p>
      <h1>{title}</h1>
    </div>
    <nav className="page-nav" aria-label="Console pages">
      <button className={!isCatalog ? "active" : ""} onClick={() => navigate("/")}>Experiments</button>
      <button className={isCatalog ? "active" : ""} onClick={() => navigate("/runs")}>Trained runs</button>
    </nav>
  </header>;
}

function RunTable({ runs, selected, onSelect, empty }: { runs: Run[]; selected: Run | null; onSelect: (run: Run) => void; empty: string }) {
  return <table>
    <thead><tr><th>Run</th><th>Scene</th><th>Status</th><th>GPU</th></tr></thead>
    <tbody>
      {runs.map(run => <tr key={run.id} onClick={() => onSelect(run)} className={selected?.id === run.id ? "selected" : ""}>
        <td>{run.id}<small>{run.preset}</small></td>
        <td>{run.scene_id}<small>{run.dataset}</small></td>
        <td><span className={`status ${run.status}`}>{run.status}</span></td>
        <td>{run.gpu_id ?? "—"}</td>
      </tr>)}
      {!runs.length && <tr className="empty-row"><td colSpan={4}>{empty}</td></tr>}
    </tbody>
  </table>;
}

function RunDetail({ selected, catalogRuns, openViewer, onRefresh }: { selected: Run | null; catalogRuns: Run[]; openViewer: (runId: string) => void; onRefresh: () => Promise<void> }) {
  const [detail, setDetail] = useState<any>(null);
  const [notice, setNotice] = useState("");
  const [tick, setTick] = useState(0);
  const [compareWith, setCompareWith] = useState("");
  const [comparison, setComparison] = useState<any>(null);

  useEffect(() => {
    setDetail(null);
    setCompareWith("");
    setComparison(null);
    if (selected) void api<any>(`/runs/${selected.id}`).then(setDetail).catch(error => setNotice(String(error)));
  }, [selected, tick]);
  useEffect(() => {
    if (!selected || selected.status === "completed") return;
    const stream = new EventSource(`/api/runs/${selected.id}/events/stream`);
    const update = () => setTick(value => value + 1);
    stream.onmessage = update;
    stream.addEventListener("pipeline", update);
    stream.addEventListener("terminal", update);
    return () => stream.close();
  }, [selected]);

  const action = async (name: "cancel" | "resume") => {
    if (!selected) return;
    try {
      await api(`/runs/${selected.id}/${name}`, { method: "POST", body: name === "resume" ? JSON.stringify({ start_at: selected.start_at, stop_after: selected.stop_after }) : undefined });
      await onRefresh();
      setTick(value => value + 1);
    } catch (error) {
      setNotice(String(error));
    }
  };
  const compare = async () => {
    if (!selected || !compareWith) return;
    try {
      setComparison(await api<any>("/compare", { method: "POST", body: JSON.stringify({ run_ids: [selected.id, compareWith] }) }));
    } catch (error) {
      setNotice(String(error));
    }
  };

  return <section className="card detail">
    <h2>Run detail</h2>
    {notice && <p className="notice">{notice}</p>}
    {!selected ? <p>Select a run to inspect it.</p> : <>
      <p><code>{selected.id}</code> · {selected.owner}</p>
      <p>{selected.dataset} / {selected.scene_id} · {selected.start_at} → {selected.stop_after}</p>
      {selected.error && <p className="error">{selected.error}</p>}
      <div className="actions">
        {["queued", "starting", "running", "cancelling"].includes(selected.status) && <button onClick={() => void action("cancel")}>Cancel</button>}
        {["failed", "cancelled", "interrupted"].includes(selected.status) && <button onClick={() => void action("resume")}>Resume</button>}
        {selected.status === "completed" && <a href={`/api/runs/${selected.id}/artifacts/run_manifest.json`} target="_blank" rel="noreferrer">Manifest</a>}
        {selected.status === "completed" && <button onClick={() => openViewer(selected.id)}>View scene</button>}
      </div>
      {selected.status === "completed" && <div className="compare">
        <select value={compareWith} onChange={event => setCompareWith(event.target.value)}>
          <option value="">Compare with completed run</option>
          {catalogRuns.filter(run => run.id !== selected.id).map(run => <option key={run.id} value={run.id}>{run.id} · {run.scene_id}</option>)}
        </select>
        <button disabled={!compareWith} onClick={() => void compare()}>Compare</button>
      </div>}
      {comparison && <section className="comparison">{comparison.runs.map((run: any) => <article key={run.id}>
        <strong>{run.id}</strong><small>{run.scene_id} · {run.preset}</small>
        <p>{run.manifest?.points?.num_points ?? "—"} points · {run.manifest?.frame_counts?.num_frames ?? "—"} frames</p>
      </article>)}</section>}
      <h3>Timeline</h3>
      <ul>{detail?.events?.map((event: any) => <li key={event.sequence}><code>{event.type}</code> <small>{event.created_at}</small></li>)}</ul>
      <h3>Artifacts</h3>
      <div className="artifacts">{detail?.artifacts?.filter((file: string) => /\.(png|jpe?g|json|zip)$/i.test(file)).slice(0, 12).map((file: string) => <a key={file} href={`/api/runs/${selected.id}/artifacts/${file}`} target="_blank" rel="noreferrer">{file}</a>)}</div>
      {selected.status !== "completed" && <><h3>Recent log</h3><Log runId={selected.id} tick={tick} /></>}
    </>}
  </section>;
}

function Log({ runId, tick }: { runId: string; tick: number }) {
  const [lines, setLines] = useState<string[]>([]);
  useEffect(() => { void api<{ lines: string[] }>(`/runs/${runId}/log`).then(result => setLines(result.lines)).catch(() => setLines([])); }, [runId, tick]);
  return <pre>{lines.join("\n") || "No output yet."}</pre>;
}

function Console({ navigate }: { navigate: Navigate }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [recoverableRuns, setRecoverableRuns] = useState<Run[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [notice, setNotice] = useState("");
  const [form, setForm] = useState({ dataset: "kitti360", scene_id: "", preset: "kitti360-dev", start_at: "prepare", stop_after: "bundle", advanced_yaml: "" });
  const refresh = async () => {
    try {
      const [active, recoverable] = await Promise.all([api<Run[]>("/runs?scope=active"), api<Run[]>("/runs?scope=recoverable")]);
      setRuns(active);
      setRecoverableRuns(recoverable);
    } catch (error) {
      setNotice(String(error));
    }
  };

  useEffect(() => {
    void refresh();
    void api<Preset[]>("/presets").then(setPresets).catch(error => setNotice(String(error)));
    void api<any[]>("/datasets").then(setDatasets).catch(error => setNotice(String(error)));
    const id = setInterval(() => void refresh(), 5000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => { if (selected && ![...runs, ...recoverableRuns].some(run => run.id === selected.id)) setSelected(null); }, [runs, recoverableRuns, selected]);

  const scenes = useMemo(() => datasets.filter(dataset => dataset.dataset === form.dataset), [datasets, form.dataset]);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    try {
      const run = await api<Run>("/runs", { method: "POST", body: JSON.stringify(form) });
      setSelected(run);
      setNotice(`Queued ${run.id}`);
      await refresh();
    } catch (error) {
      setNotice(String(error));
    }
  };

  return <main>
    <Header title="Experiments" navigate={navigate} />
    {notice && <p className="notice">{notice}</p>}
    <section className="grid">
      <aside className="card"><h2>New run</h2><form onSubmit={submit}>
        <label>Dataset<select value={form.dataset} onChange={event => setForm({ ...form, dataset: event.target.value, preset: event.target.value === "nvidia_ncore" ? "ncore-dev" : "kitti360-dev" })}><option value="kitti360">KITTI-360</option><option value="nvidia_ncore">NVIDIA NCore</option></select></label>
        <label>Mounted scene<select required value={form.scene_id} onChange={event => setForm({ ...form, scene_id: event.target.value })}><option value="">Select a discovered scene</option>{scenes.map(scene => <option key={scene.scene_id} value={scene.scene_id}>{scene.scene_id} — {scene.status}</option>)}</select></label>
        <label>Recipe<select value={form.preset} onChange={event => setForm({ ...form, preset: event.target.value })}>{presets.filter(preset => preset.datasets.includes(form.dataset)).map(preset => <option key={preset.slug} value={preset.slug}>{preset.name}</option>)}</select></label>
        <div className="two"><label>Start<select value={form.start_at} onChange={event => setForm({ ...form, start_at: event.target.value })}>{stages.map(stage => <option key={stage}>{stage}</option>)}</select></label><label>Stop<select value={form.stop_after} onChange={event => setForm({ ...form, stop_after: event.target.value })}>{stages.map(stage => <option key={stage}>{stage}</option>)}</select></label></div>
        <label>Advanced safe YAML<textarea placeholder={"train:\n  iterations: 30000"} value={form.advanced_yaml} onChange={event => setForm({ ...form, advanced_yaml: event.target.value })} /></label>
        <button className="primary">Queue run</button>
      </form>{recoverableRuns.length > 0 && <section className="attention"><h2>Needs attention</h2><p className="muted">Stopped runs can be resumed.</p><div className="attention-list">{recoverableRuns.map(run => <button key={run.id} onClick={() => setSelected(run)} className={selected?.id === run.id ? "selected" : ""}>{run.id}<small><span className={`status ${run.status}`}>{run.status}</span> · {run.scene_id}</small></button>)}</div></section>}</aside>
      <section className="card runs"><div className="card-heading"><div><h2>Active queue</h2><p className="muted">Queued and in-progress runs only.</p></div><button onClick={() => void refresh()}>Refresh</button></div><RunTable runs={runs} selected={selected} onSelect={setSelected} empty="No queued or active runs." /></section>
      <RunDetail selected={selected} catalogRuns={[]} openViewer={runId => navigate(`/runs/${encodeURIComponent(runId)}/viewer`)} onRefresh={refresh} />
    </section>
  </main>;
}

function CompletedRuns({ navigate }: { navigate: Navigate }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [notice, setNotice] = useState("");
  const refresh = async () => { try { setRuns(await api<Run[]>("/runs?scope=completed")); } catch (error) { setNotice(String(error)); } };
  useEffect(() => { void refresh(); }, []);

  return <main>
    <Header title="Trained runs" navigate={navigate} />
    {notice && <p className="notice">{notice}</p>}
    <section className="catalog-grid">
      <section className="card runs"><div className="card-heading"><div><h2>Completed training runs</h2><p className="muted">Browse finished VBOGS scenes and outputs.</p></div><button onClick={() => void refresh()}>Refresh</button></div><RunTable runs={runs} selected={selected} onSelect={setSelected} empty="No completed training runs yet." /></section>
      <RunDetail selected={selected} catalogRuns={runs} openViewer={runId => navigate(`/runs/${encodeURIComponent(runId)}/viewer`)} onRefresh={refresh} />
    </section>
  </main>;
}

function Root() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => { const pop = () => setPath(window.location.pathname); window.addEventListener("popstate", pop); return () => window.removeEventListener("popstate", pop); }, []);
  const navigate: Navigate = next => { window.history.pushState({}, "", next); setPath(next); };
  const viewerRoute = path.match(/^\/runs\/([^/]+)\/viewer\/?$/);
  if (viewerRoute) return <SceneViewer runId={decodeURIComponent(viewerRoute[1])} onBack={() => navigate("/runs")} />;
  if (path === "/runs" || path === "/runs/") return <CompletedRuns navigate={navigate} />;
  return <Console navigate={navigate} />;
}

createRoot(document.getElementById("root")!).render(<Root />);
