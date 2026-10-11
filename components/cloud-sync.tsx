"use client";
import {useEffect,useRef,useState} from 'react';
import {validateWorkspace,type Workspace} from '../lib/workspace';
import type {WorkspaceRow} from '../lib/workspace-store';
export function readWorkspace(key:string):Workspace{
 return validateWorkspace({portfolio:JSON.parse(localStorage.getItem(key)||'{"holdings":[],"snapshots":[]}'),watchlist:JSON.parse(localStorage.getItem(key+'-watchlist')||'null'),flows:JSON.parse(localStorage.getItem(key+'-flows')||'[]'),targets:JSON.parse(localStorage.getItem(key+'-targets')||'null'),theme:localStorage.getItem('folio-theme'),...(localStorage.getItem(key+'-layouts')?{layouts:JSON.parse(localStorage.getItem(key+'-layouts')!)}:{}),...(localStorage.getItem(key+'-research')?{research:JSON.parse(localStorage.getItem(key+'-research')!)}:{})});
}
export function applyWorkspace(key:string,w:Workspace){
 localStorage.setItem(key,JSON.stringify(w.portfolio));
 for(const [suffix,value]of [['-watchlist',w.watchlist],['-flows',w.flows],['-targets',w.targets]] as const){if(value===null)localStorage.removeItem(key+suffix);else localStorage.setItem(key+suffix,JSON.stringify(value));}
 if(w.layouts)localStorage.setItem(key+'-layouts',JSON.stringify(w.layouts));else localStorage.removeItem(key+'-layouts');
 if(w.research)localStorage.setItem(key+'-research',JSON.stringify(w.research));else localStorage.removeItem(key+'-research');
 if(w.theme)localStorage.setItem('folio-theme',w.theme);
}
export function useCloudSync(key:string,onLoad:(w:Workspace)=>void){
 const [status,setStatus]=useState('Checking cloud storage…'),[mode,setMode]=useState<'checking'|'local'|'synced'|'pending'|'error'|'conflict'>('checking');
 const onLoadRef=useRef(onLoad);onLoadRef.current=onLoad;
 const control=useRef<{revision:number|null;hash:string;configured:boolean;blocked:boolean;running:boolean;queued:boolean;disposed:boolean}>({revision:null,hash:'',configured:false,blocked:false,running:false,queued:false,disposed:false});
 const initializeRef=useRef<(force?:boolean)=>Promise<void>>(async()=>{});
 useEffect(()=>{
  const c={revision:null as number|null,hash:'',configured:false,blocked:false,running:false,queued:false,disposed:false};control.current=c;
  let timer:ReturnType<typeof setTimeout>|undefined;
  const mark=(state:typeof mode,message:string)=>{if(!c.disposed){setMode(state);setStatus(message);}};
  const remember=()=>localStorage.setItem(key+'-cloud-meta',JSON.stringify({revision:c.revision,hash:c.hash}));
  async function push(){
   if(c.disposed||!c.configured||c.blocked)return;
   if(c.running){c.queued=true;return;}
   c.running=true;
   try{
    const data=readWorkspace(key),hash=JSON.stringify(data);
    if(hash===c.hash){mark('synced','Saved to Supabase');return;}
    mark('pending','Saving to Supabase…');
    const response=await fetch('/api/workspace',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:c.revision,data}),signal:AbortSignal.timeout(20000)});
    const body=await response.json() as {row?:WorkspaceRow;error?:string};if(c.disposed)return;
    if(response.status===409){c.blocked=true;mark('conflict','Another device has newer data. Your local copy is kept.');return;}
    if(!response.ok||!body.row)throw Error(body.error||'Cloud save failed.');
    c.revision=body.row.revision;c.hash=hash;remember();mark('synced','Saved to Supabase');
   }catch(error){mark('error',(error as Error).message+' Your local copy is kept.');}
   finally{c.running=false;if(c.queued){c.queued=false;void push();}}
  }
  async function initialize(force=false){
   if(c.running)return;
   c.running=true;c.blocked=true;mark('checking','Checking cloud storage…');
   try{
    const response=await fetch('/api/workspace',{cache:'no-store',signal:AbortSignal.timeout(20000)});const body=await response.json() as {configured?:boolean;row?:WorkspaceRow|null;error?:string};if(c.disposed)return;
    if(!response.ok)throw Error(body.error||'Cloud storage is unavailable.');
    if(!body.configured){c.configured=false;mark('local','Saved on this device · cloud storage not configured');return;}
    c.configured=true;
    const local=readWorkspace(key),hash=JSON.stringify(local),row=body.row as WorkspaceRow|null;
    const meta=JSON.parse(localStorage.getItem(key+'-cloud-meta')||'null');
    if(row){
     const data=validateWorkspace(row.data),cloudHash=JSON.stringify(data);
     const hasLocal=localStorage.getItem(key)!==null;
     const dirty=hasLocal&&meta?.hash!==hash;
     if(!force&&hash!==cloudHash&&dirty&&meta?.revision!==row.revision){mark('conflict','Local and cloud data differ. Export your local copy, then choose which to keep.');return;}
     c.revision=row.revision;c.hash=cloudHash;
     if(force||!hasLocal||(!dirty&&hash!==cloudHash)){applyWorkspace(key,data);onLoadRef.current(data);}
     remember();c.blocked=false;mark('synced','Saved to Supabase');
    }else{c.revision=null;c.hash='';c.blocked=false;mark('pending','Ready to save this workspace to Supabase');}
   }catch(error){mark('error',(error as Error).message+' Your local copy is kept.');}
   finally{c.running=false;if(c.configured&&!c.blocked)void push();}
  }
  initializeRef.current=initialize;void initialize();
  const changed=(event:Event)=>{const changedKey=(event as CustomEvent<{key:string}>).detail?.key;if(changedKey===key||changedKey?.startsWith(key+'-')||changedKey==='folio-theme'){clearTimeout(timer);timer=setTimeout(()=>void push(),800);}};
  const focus=()=>{if(!c.running&&!c.blocked)void initialize();};
  window.addEventListener('folio-local-change',changed);window.addEventListener('online',focus);window.addEventListener('focus',focus);
  return()=>{c.disposed=true;clearTimeout(timer);window.removeEventListener('folio-local-change',changed);window.removeEventListener('online',focus);window.removeEventListener('focus',focus);};
 },[key]);
 return {status,mode,retry:()=>void initializeRef.current(),loadCloud:()=>void initializeRef.current(true)};
}
