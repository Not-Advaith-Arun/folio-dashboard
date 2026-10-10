import { validateWorkspace, type Workspace } from './workspace.ts';
export class WorkspaceError extends Error { status: number; constructor(message:string,status=503){super(message);this.status=status;} }
export type WorkspaceRow = {data:Workspace;revision:number;updated_at:string};
export function workspaceStore(url:string,key:string,transport:typeof fetch=fetch) {
 const base = new URL(url);
 if(base.protocol!=='https:' || !base.hostname.endsWith('.supabase.co') || base.pathname!=='/' || base.search || base.hash || base.username || base.password) throw new WorkspaceError('Invalid database configuration.');
 if(!key.startsWith('sb_secret_') && !key.startsWith('eyJ')) throw new WorkspaceError('Server database key is required.');
 async function request(userId:string,method:string,revision?:number,data?:Workspace):Promise<WorkspaceRow|null>{
  if(!userId || userId.length>250)throw new WorkspaceError('Sign in to access your workspace.',401);
  const endpoint=new URL('/rest/v1/folio_workspaces',base);
  if(method!=='POST') endpoint.searchParams.set('user_id','eq.'+userId);
  endpoint.searchParams.set('select','data,revision,updated_at');
  if(method==='PATCH')endpoint.searchParams.set('revision','eq.'+revision);
  const headers:Record<string,string>={apikey:key,'Content-Type':'application/json',Prefer:'return=representation'};
  if(key.startsWith('eyJ'))headers.Authorization='Bearer '+key;
  const body=data?JSON.stringify({...(method==='POST'?{user_id:userId}:{}),data,revision:(revision??0)+1,updated_at:new Date().toISOString()}):undefined;
  let response:Response;try{response=await transport(endpoint,{method,headers,body,signal:AbortSignal.timeout(15000)});}catch{throw new WorkspaceError('Cloud storage is unavailable. Your local copy is kept.');}
  if(response.status===409)throw new WorkspaceError('Your cloud workspace changed on another device.',409);
  if(!response.ok)throw new WorkspaceError('Cloud storage is unavailable. Your local copy is kept.');
  const rows=await response.json() as WorkspaceRow[];
  if(!Array.isArray(rows)||rows.length>1)throw new WorkspaceError('Unexpected cloud response.');
  if(!rows.length){if(method==='PATCH')throw new WorkspaceError('Your cloud workspace changed on another device.',409);return null;}
  const row=rows[0];if(!Number.isSafeInteger(row.revision)||row.revision<1||typeof row.updated_at!=='string')throw new WorkspaceError('Invalid cloud response.');
  return {data:validateWorkspace(row.data),revision:row.revision,updated_at:row.updated_at};
 }
 return {read:(userId:string)=>request(userId,'GET'),write:(userId:string,data:unknown,revision:number|null)=>{
  if(revision!==null&&(!Number.isSafeInteger(revision)||revision<1||revision>=2147483647))throw new WorkspaceError('Invalid workspace version.',400);
  return request(userId,revision===null?'POST':'PATCH',revision??undefined,validateWorkspace(data));
 }};
}
