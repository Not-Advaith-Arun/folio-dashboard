import { getChatGPTUser } from '../../chatgpt-auth';
import { workspaceStore, WorkspaceError } from '../../../lib/workspace-store';
export const dynamic='force-dynamic';
const respond=(data:unknown,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store'}});
function store(){const url=process.env.SUPABASE_URL,key=process.env.SUPABASE_SECRET_KEY;return url&&key?workspaceStore(url,key):null;}
export async function GET(){
 const user=await getChatGPTUser();if(!user)return respond({error:'Sign in to access your workspace.'},401);
 try{const db=store();return db?respond({configured:true,row:await db.read(user.userId)}):respond({configured:false,row:null});}
 catch(error){return respond({error:error instanceof WorkspaceError?error.message:'Cloud storage is unavailable.'},503);}
}
export async function PUT(request:Request){
 const user=await getChatGPTUser();if(!user)return respond({error:'Sign in to access your workspace.'},401);
 if(request.headers.get('origin')!==new URL(request.url).origin)return respond({error:'Request must come from this app.'},403);
 if(Number(request.headers.get('content-length')||0)>5000000)return respond({error:'Workspace is too large.'},413);
 try{const db=store();if(!db)return respond({error:'Cloud storage is not configured.'},503);const text=await request.text();if(text.length>5000000)return respond({error:'Workspace is too large.'},413);const body=JSON.parse(text);if(!Object.hasOwn(body,'revision'))return respond({error:'Workspace version is required.'},400);return respond({configured:true,row:await db.write(user.userId,body.data,body.revision)});}
 catch(error){return respond({error:error instanceof WorkspaceError?error.message:'Invalid workspace data.'},error instanceof WorkspaceError?error.status:400);}
}
