import React, { useCallback, useEffect, useRef, useState } from "react";

type Matrix = number[][];
type Camera = { id:string; image_name:string; c2w:Matrix; source:string };
type Session = { run_id?:string|null; gpu_id?:string|null; owner?:string|null; status?:string; revision?:number };
type Metadata = { drive:string; anchor_count:number; max_fps:number; jpeg_quality:number; rgb_only?:boolean; render_modes?:string[] };
type Readiness = { ready:boolean; reason?:string|null; source?:string|null };

const modes = ["side_by_side", "rgb", "uncertainty", "alpha"] as const;
type Mode = typeof modes[number];

async function api<T>(path:string, options?:RequestInit):Promise<T>{
  const response=await fetch(`/api${path}`,{headers:{"Content-Type":"application/json",...(options?.headers||{})},...options});
  if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof body.detail==="string"?body.detail:(body.detail?.message||response.statusText));}
  return response.json();
}

function cloneMatrix(matrix:Matrix):Matrix{return matrix.map(row=>row.slice());}
function identity4():Matrix{return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];}
function mat3Mul(a:number[][],b:number[][]):number[][]{return Array.from({length:3},(_,r)=>Array.from({length:3},(_,c)=>a[r][0]*b[0][c]+a[r][1]*b[1][c]+a[r][2]*b[2][c]));}
function rot(axis:number[],angle:number):number[][]{const[x,y,z]=axis,c=Math.cos(angle),s=Math.sin(angle),t=1-c;return [[t*x*x+c,t*x*y-s*z,t*x*z+s*y],[t*x*y+s*z,t*y*y+c,t*y*z-s*x],[t*x*z-s*y,t*y*z+s*x,t*z*z+c]];}
function normalize(v:number[]):number[]{const n=Math.hypot(...v)||1;return v.map(value=>value/n);}
function rotation(matrix:Matrix):number[][]{return matrix.slice(0,3).map(row=>row.slice(0,3));}
function setRotation(matrix:Matrix, value:number[][]):Matrix{const next=cloneMatrix(matrix);for(let r=0;r<3;r++)for(let c=0;c<3;c++)next[r][c]=value[r][c];return next;}
function yprToC2w(values:number[]):Matrix{const [x,y,z,yawDeg,pitchDeg,rollDeg]=values;if(!values.every(Number.isFinite))throw new Error("Fill all pose fields with numbers");const yaw=rot([0,0,1],yawDeg*Math.PI/180),pitch=rot([0,1,0],pitchDeg*Math.PI/180),roll=rot([1,0,0],rollDeg*Math.PI/180);const next=setRotation(identity4(),mat3Mul(yaw,mat3Mul(pitch,roll)));next[0][3]=x;next[1][3]=y;next[2][3]=z;return next;}
function c2wToPose(matrix:Matrix):string[]{const r=rotation(matrix),pitch=Math.asin(Math.max(-1,Math.min(1,-r[2][0])));const yaw=Math.abs(Math.cos(pitch))>1e-6?Math.atan2(r[1][0],r[0][0]):Math.atan2(-r[0][1],r[1][1]);const roll=Math.abs(Math.cos(pitch))>1e-6?Math.atan2(r[2][1],r[2][2]):0;return [matrix[0][3],matrix[1][3],matrix[2][3],yaw*180/Math.PI,pitch*180/Math.PI,roll*180/Math.PI].map(value=>Number(value.toFixed(6)).toString());}
function wsUrl():string{const protocol=window.location.protocol==="https:"?"wss":"ws";return `${protocol}://${window.location.host}/api/viewer/ws/render`;}

export function SceneViewer({runId,onBack}:{runId:string;onBack:()=>void}){
  const [readiness,setReadiness]=useState<Readiness|null>(null);
  const [session,setSession]=useState<Session|null>(null);
  const [metadata,setMetadata]=useState<Metadata|null>(null);
  const [cameras,setCameras]=useState<Camera[]>([]);
  const [cameraId,setCameraId]=useState("");
  const [c2w,setC2w]=useState<Matrix|null>(null);
  const [layer,setLayer]=useState<Mode>("side_by_side");
  const [pose,setPose]=useState<string[]>(["","","","","",""]);
  const [status,setStatus]=useState("Checking viewer availability…");
  const [notice,setNotice]=useState("");
  const [imageUrl,setImageUrl]=useState<string|null>(null);
  const socketRef=useRef<WebSocket|null>(null);
  const c2wRef=useRef<Matrix|null>(null);
  const keysRef=useRef(new Set<string>());
  const requestId=useRef(0);
  const [reconnectEpoch,setReconnectEpoch]=useState(0);
  const [connectedEpoch,setConnectedEpoch]=useState(0);
  const [loadEpoch,setLoadEpoch]=useState(0);

  const refreshSession=useCallback(async()=>{const [nextReadiness,nextSession]=await Promise.all([api<Readiness>(`/runs/${runId}/viewer-readiness`),api<Session>("/viewer")]);setReadiness(nextReadiness);setSession(nextSession);return nextSession;},[runId]);
  const loadRenderer=useCallback(async()=>{try{setStatus("Loading scene metadata…");const [nextMetadata, payload]=await Promise.all([api<Metadata>("/viewer/metadata"),api<{default_camera_id:string;cameras:Record<string,Camera[]>}>("/viewer/cameras")]);const nextCameras=[...(payload.cameras.train||[]),...(payload.cameras.test||[])];const first=nextCameras.find(camera=>camera.id===payload.default_camera_id)||nextCameras[0];if(!first)throw new Error("The loaded scene has no render cameras");setMetadata(nextMetadata);setCameras(nextCameras);setCameraId(first.id);setC2w(cloneMatrix(first.c2w));setPose(c2wToPose(first.c2w));setLayer(nextMetadata.rgb_only?"rgb":"side_by_side");setStatus("Connecting to renderer…");}catch{setStatus("Waiting for the GPU renderer to load…");window.setTimeout(()=>setLoadEpoch(value=>value+1),1500);}},[]);

  useEffect(()=>{void refreshSession().catch(error=>setNotice(String(error)));const timer=window.setInterval(()=>void refreshSession().catch(()=>undefined),5000);return()=>window.clearInterval(timer);},[refreshSession]);
  useEffect(()=>{if(session?.run_id===runId&&session.status==="active")void loadRenderer();else {setMetadata(null);setCameras([]);setC2w(null);}},[session?.run_id,session?.revision,session?.status,runId,loadRenderer,loadEpoch]);
  useEffect(()=>{c2wRef.current=c2w;if(c2w)setPose(c2wToPose(c2w));},[c2w]);
  useEffect(()=>()=>{if(imageUrl)URL.revokeObjectURL(imageUrl);},[imageUrl]);

  const loadRun=async()=>{if(!readiness?.ready)return;const replacing=Boolean(session?.run_id&&session.run_id!==runId);if(replacing&&!window.confirm(`Replace the active viewer for ${session?.run_id}? This stops its shared GPU scene.`))return;try{setStatus("Starting GPU renderer…");const next=await api<Session>("/viewer",{method:"POST",body:JSON.stringify({run_id:runId,confirm_replace:replacing})});setSession(next);setNotice("");}catch(error){setStatus(String(error));}};
  const stop=async()=>{if(!window.confirm("Stop the shared renderer and release its GPU slot?"))return;try{setSession(await api<Session>("/viewer",{method:"DELETE"}));setStatus("Viewer stopped");}catch(error){setNotice(String(error));}};

  useEffect(()=>{if(!metadata||session?.run_id!==runId)return;let disposed=false;const socket=new WebSocket(wsUrl());socket.binaryType="blob";socketRef.current=socket;socket.onopen=()=>{setStatus("Connected");setConnectedEpoch(value=>value+1);};socket.onclose=()=>{if(!disposed){setStatus("Disconnected — retrying…");window.setTimeout(()=>setReconnectEpoch(value=>value+1),1000);}};socket.onerror=()=>setStatus("Renderer connection failed");socket.onmessage=event=>{if(typeof event.data==="string"){try{const message=JSON.parse(event.data);setStatus(message.error||`${message.mode||"render"} ${Number(message.elapsed_ms||0).toFixed(1)} ms`);}catch{setStatus(event.data);}}else{const url=URL.createObjectURL(event.data as Blob);setImageUrl(previous=>{if(previous)URL.revokeObjectURL(previous);return url;});}};return()=>{disposed=true;socket.close();if(socketRef.current===socket)socketRef.current=null;};},[metadata,session?.run_id,runId,reconnectEpoch]);
  useEffect(()=>{if(!c2w||!cameraId||!metadata)return;const timer=window.setTimeout(()=>{const socket=socketRef.current;if(socket?.readyState===WebSocket.OPEN)socket.send(JSON.stringify({request_id:++requestId.current,camera_id:cameraId,c2w,layer,quality:metadata.jpeg_quality||85}));},50);return()=>window.clearTimeout(timer);},[c2w,cameraId,layer,metadata,connectedEpoch]);

  useEffect(()=>{const move=window.setInterval(()=>{const matrix=c2wRef.current;if(!matrix||keysRef.current.size===0)return;const r=rotation(matrix),right=normalize([r[0][0],r[1][0],r[2][0]]),forward=normalize([r[0][2],r[1][2],r[2][2]]),step=keysRef.current.has("shift")?1:0.35;let next=cloneMatrix(matrix);const translate=(axis:number[],amount:number)=>{for(let i=0;i<3;i++)next[i][3]+=axis[i]*amount;};if(keysRef.current.has("w"))translate(forward,step);if(keysRef.current.has("s"))translate(forward,-step);if(keysRef.current.has("a"))translate(right,-step);if(keysRef.current.has("d"))translate(right,step);if(keysRef.current.has("q"))translate([0,0,1],-step);if(keysRef.current.has("e"))translate([0,0,1],step);setC2w(next);},50);const down=(event:KeyboardEvent)=>{if(["INPUT","SELECT","TEXTAREA","BUTTON"].includes((event.target as HTMLElement)?.tagName))return;const key=event.key.toLowerCase();if(["w","a","s","d","q","e"].includes(key)){event.preventDefault();keysRef.current.add(key);}if(event.key==="Shift")keysRef.current.add("shift");};const up=(event:KeyboardEvent)=>{keysRef.current.delete(event.key.toLowerCase());if(event.key==="Shift")keysRef.current.delete("shift");};window.addEventListener("keydown",down);window.addEventListener("keyup",up);return()=>{window.clearInterval(move);window.removeEventListener("keydown",down);window.removeEventListener("keyup",up);};},[]);

  const selectCamera=(id:string)=>{const camera=cameras.find(value=>value.id===id);if(!camera)return;setCameraId(id);setC2w(cloneMatrix(camera.c2w));};
  const pointer=useRef<{x:number;y:number}|null>(null);
  const rotateCamera=(event:React.PointerEvent<HTMLDivElement>)=>{if(!pointer.current||!c2wRef.current)return;const dx=event.clientX-pointer.current.x,dy=event.clientY-pointer.current.y;pointer.current={x:event.clientX,y:event.clientY};const current=rotation(c2wRef.current),yaw=rot([0,0,1],-dx*.004),right=normalize([current[0][0],current[1][0],current[2][0]]),pitch=rot(right,-dy*.003);setC2w(setRotation(c2wRef.current,mat3Mul(pitch,mat3Mul(yaw,current))));};
  const allowedModes=metadata?.rgb_only?["rgb"]:modes;

  return <main className="viewer-page"><header className="viewer-header"><button onClick={onBack}>← Run details</button><div><p className="eyebrow">VBOGS SCENE VIEWER</p><h1>{runId}</h1></div><div className="viewer-session">{session?.run_id&&session.status==="active"?<>Active: <code>{session.run_id}</code> · GPU {session.gpu_id}</>:"No active scene"}</div></header>
    {notice&&<p className="notice">{notice}</p>}
    {!readiness?<section className="viewer-empty">Checking render artifacts…</section>:!readiness.ready?<section className="viewer-empty"><h2>Scene viewer unavailable</h2><p>{readiness.reason}</p></section>:session?.run_id!==runId||session.status!=="active"?<section className="viewer-empty"><h2>Load this scene</h2><p>{session?.run_id&&session.status==="active"?`Loading this run will replace the shared viewer for ${session.run_id}.`:"The trained scene and U.npy are ready to render."}</p><button className="primary" onClick={()=>void loadRun()}>Load scene viewer</button></section>:<section className="scene-shell"><div className="scene-canvas" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);pointer.current={x:event.clientX,y:event.clientY};}} onPointerMove={rotateCamera} onPointerUp={()=>{pointer.current=null;}}>{imageUrl?<img src={imageUrl} alt="Rendered VBOGS scene"/>:<p>{status}</p>}</div><aside className="scene-controls"><label>Saved camera<select value={cameraId} onChange={event=>selectCamera(event.target.value)}>{cameras.map(camera=><option key={camera.id} value={camera.id}>{camera.id} · {camera.image_name}</option>)}</select></label><button onClick={()=>selectCamera(cameraId)}>Reset saved pose</button><div className="mode-buttons">{allowedModes.map(mode=><button key={mode} className={layer===mode?"active":""} onClick={()=>setLayer(mode as Mode)}>{mode==="side_by_side"?"Split":mode}</button>)}</div><h2>Teleport</h2><div className="pose-grid">{["x","y","z","yaw","pitch","roll"].map((name,index)=><input key={name} aria-label={name} value={pose[index]} onChange={event=>setPose(values=>values.map((value,i)=>i===index?event.target.value:value))} onKeyDown={event=>{if(event.key==="Enter"){try{setC2w(yprToC2w(pose.map(Number)));}catch(error){setNotice(String(error));}}}}/>)}</div><button onClick={()=>{try{setC2w(yprToC2w(pose.map(Number)));setNotice("");}catch(error){setNotice(String(error));}}}>Go to pose</button><p className="viewer-help">Drag to look · WASD move · Q/E vertical · Shift faster</p><p className="viewer-status">{status}</p><button className="danger" onClick={()=>void stop()}>Stop shared viewer</button></aside></section>}
  </main>;
}
