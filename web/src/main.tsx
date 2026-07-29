import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import "./ncore.css";
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
type NCoreClip = { scene_id: string; status: "missing" | "partial" | "ready" };
type NCoreDownload = { id: string; scene_id: string; owner: string; status: string; error?: string; created_at?: string };
type Progress = {
  state: string;
  status: string;
  overall: { completed_stages: number; total_stages: number; percent: number };
  current_stage: { name: string; index: number; total: number } | null;
  training: { state: string; current_iterations: number; total_iterations: number; updated_at?: string } | null;
};
type Navigate = (path: string) => void;

const stages = ["prepare", "train", "stereo", "bucket", "fit", "inspect", "uncertainty", "map-viz", "render", "nbv", "nbv-viz", "bundle"];
const deletableStatuses = new Set(["queued", "cancelled", "completed", "failed", "interrupted"]);

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

function ProgressPanel({ progress }: { progress?: Progress }) {
  if (!progress) return null;
  const percent = Math.max(0, Math.min(100, progress.overall.percent));
  const trainingPercent = progress.training
    ? Math.round((progress.training.current_iterations / progress.training.total_iterations) * 1000) / 10
    : 0;
  const stateLabel: Record<string, string> = {
    queued: "Waiting for a GPU slot", starting: "Starting pipeline", running: "Running",
    cancelling: "Cancellation requested", cancelled: "Run cancelled", failed: "Run failed",
    interrupted: "Run interrupted", completed: "Run completed",
    finalizing: "Training iterations complete; rendering and evaluating",
  };
  const stageLabel = progress.current_stage
    ? `Stage ${progress.current_stage.index} of ${progress.current_stage.total}: ${progress.current_stage.name}`
    : progress.status === "completed" ? "All selected stages completed" : "No stage is running";

  return <section className="progress-panel" aria-live="polite">
    <div className="progress-heading"><h3>Progress</h3><strong>{percent}%</strong></div>
    <p className="muted">{stateLabel[progress.state] || progress.state} · {stageLabel}</p>
    <div className="progress-track" role="progressbar" aria-label="Overall pipeline progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
      <div className="progress-fill" style={{ width: `${percent}%` }} />
    </div>
    <small>{progress.overall.completed_stages} of {progress.overall.total_stages} stages completed</small>
    {progress.training && <div className="training-progress">
      <div className="progress-heading"><span>Training</span><strong>{trainingPercent}%</strong></div>
      <div className="progress-track" role="progressbar" aria-label="Training iteration progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={trainingPercent}>
        <div className="progress-fill training" style={{ width: `${trainingPercent}%` }} />
      </div>
      <small>{progress.training.current_iterations.toLocaleString()} / {progress.training.total_iterations.toLocaleString()} iterations{progress.training.state === "finalizing" ? " · finalizing" : ""}</small>
    </div>}
  </section>;
}

function RunDetail({ selected, catalogRuns, openViewer, onRefresh, onDeleted }: { selected: Run | null; catalogRuns: Run[]; openViewer: (runId: string) => void; onRefresh: () => Promise<void>; onDeleted: () => void }) {
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
    stream.addEventListener("progress", update);
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
  const deleteRun = async () => {
    if (!selected) return;
    const confirmation = window.prompt(`This permanently deletes ${selected.id} and all of its files. Type the run ID to continue:`);
    if (confirmation === null) return;
    if (confirmation !== selected.id) {
      setNotice("Run ID did not match; nothing was deleted.");
      return;
    }
    try {
      await api<{ id: string; deleted: boolean }>(`/runs/${selected.id}`, { method: "DELETE", body: JSON.stringify({ confirm_run_id: confirmation }) });
      onDeleted();
      await onRefresh();
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
      <ProgressPanel progress={detail?.progress} />
      {selected.error && <p className="error">{selected.error}</p>}
      <div className="actions">
        {["queued", "starting", "running", "cancelling"].includes(selected.status) && <button onClick={() => void action("cancel")}>Cancel</button>}
        {["failed", "cancelled", "interrupted"].includes(selected.status) && <button onClick={() => void action("resume")}>Resume</button>}
        {selected.status === "completed" && <a href={`/api/runs/${selected.id}/artifacts/run_manifest.json`} target="_blank" rel="noreferrer">Manifest</a>}
        {selected.status === "completed" && <button onClick={() => openViewer(selected.id)}>View scene</button>}
        {deletableStatuses.has(selected.status) && <button className="danger" onClick={() => void deleteRun()}>Delete run</button>}
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

function NCoreDownloads({ onDatasetRefresh }: { onDatasetRefresh: () => Promise<void> }) {
  const [query, setQuery] = useState("");
  const [clips, setClips] = useState<NCoreClip[]>([]);
  const [selected, setSelected] = useState("");
  const [downloads, setDownloads] = useState<NCoreDownload[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [notice, setNotice] = useState("");

  const refreshDownloads = async () => {
    try {
      const jobs = await api<NCoreDownload[]>("/ncore/downloads");
      setDownloads(jobs);
      const active = jobs.find(job => job.status === "running") || jobs[0];
      if (active) setEvents((await api<{ events: any[] }>(`/ncore/downloads/${active.id}/log`)).events);
      else setEvents([]);
      if (jobs.some(job => job.status === "completed")) await onDatasetRefresh();
    } catch (error) { setNotice(String(error)); }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const params = new URLSearchParams({ query, limit: "100" });
      void api<{ clips: NCoreClip[] }>(`/ncore/catalog?${params}`).then(result => {
        setClips(result.clips);
        setNotice("");
      }).catch(error => setNotice(String(error)));
    }, 200);
    return () => window.clearTimeout(timer);
  }, [query]);
  useEffect(() => {
    void refreshDownloads();
    const timer = window.setInterval(() => void refreshDownloads(), 2500);
    return () => window.clearInterval(timer);
  }, []);

  const start = async () => {
    if (!selected) return;
    try {
      const job = await api<NCoreDownload>("/ncore/downloads", { method: "POST", body: JSON.stringify({ scene_id: selected }) });
      setNotice(`Queued ${job.scene_id}`);
      setSelected("");
      await refreshDownloads();
    } catch (error) { setNotice(String(error)); }
  };
  const active = downloads.some(job => ["queued", "running"].includes(job.status));

  return <section className="ncore-downloads">
    <h2>NCore downloads</h2>
    <p className="muted">Select one authorized clip. The server downloads all camera, LiDAR, and core components needed for reconstruction.</p>
    {notice && <p className="notice">{notice}</p>}
    <label>Search NCore catalog<input value={query} onChange={event => setQuery(event.target.value)} placeholder="Clip UUID" /></label>
    <label>Available clip<select value={selected} onChange={event => setSelected(event.target.value)}>
      <option value="">Select a missing or partial clip</option>
      {clips.map(clip => <option key={clip.scene_id} value={clip.scene_id} disabled={clip.status === "ready"}>{clip.scene_id} — {clip.status}</option>)}
    </select></label>
    <button className="primary" type="button" disabled={!selected || active} onClick={() => void start()}>{active ? "Download in progress" : "Download full clip"}</button>
    <div className="download-history">
      {downloads.slice(0, 4).map(job => <p key={job.id}><span className={`status ${job.status}`}>{job.status}</span> {job.scene_id}{job.error && <small className="error">{job.error}</small>}</p>)}
    </div>
    {events.length > 0 && <pre className="download-log">{events.map(event => event.message).join("\n")}</pre>}
  </section>;
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
  const refreshDatasets = async () => {
    try { setDatasets(await api<any[]>("/datasets")); } catch (error) { setNotice(String(error)); }
  };

  useEffect(() => {
    void refresh();
    void api<Preset[]>("/presets").then(setPresets).catch(error => setNotice(String(error)));
    void refreshDatasets();
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
      </form>{form.dataset === "nvidia_ncore" && <NCoreDownloads onDatasetRefresh={refreshDatasets} />}{recoverableRuns.length > 0 && <section className="attention"><h2>Needs attention</h2><p className="muted">Stopped runs can be resumed.</p><div className="attention-list">{recoverableRuns.map(run => <button key={run.id} onClick={() => setSelected(run)} className={selected?.id === run.id ? "selected" : ""}>{run.id}<small><span className={`status ${run.status}`}>{run.status}</span> · {run.scene_id}</small></button>)}</div></section>}</aside>
      <section className="card runs"><div className="card-heading"><div><h2>Active queue</h2><p className="muted">Queued and in-progress runs only.</p></div><button onClick={() => void refresh()}>Refresh</button></div><RunTable runs={runs} selected={selected} onSelect={setSelected} empty="No queued or active runs." /></section>
      <RunDetail selected={selected} catalogRuns={[]} openViewer={runId => navigate(`/runs/${encodeURIComponent(runId)}/viewer`)} onRefresh={refresh} onDeleted={() => setSelected(null)} />
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
      <RunDetail selected={selected} catalogRuns={runs} openViewer={runId => navigate(`/runs/${encodeURIComponent(runId)}/viewer`)} onRefresh={refresh} onDeleted={() => setSelected(null)} />
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
