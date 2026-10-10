import {getChatGPTUser} from '../../chatgpt-auth';
export const dynamic='force-dynamic';
const cache=new Map<string,{time:number;body:unknown}>();
export async function GET(request:Request){
 const headers={'Cache-Control':'no-store'};
 if(!await getChatGPTUser())return Response.json({error:'Sign in to explore funds.'},{status:401,headers});
 const code=new URL(request.url).searchParams.get('code')||'';if(!/^\d{4,8}$/.test(code))return Response.json({error:'Choose a valid AMFI scheme code.'},{status:400,headers});
 try{const saved=cache.get(code);if(saved&&Date.now()-saved.time<3600000)return Response.json(saved.body,{headers});const r=await fetch(`https://api.mfapi.in/mf/${code}`,{signal:AbortSignal.timeout(15000)});if(!r.ok)throw new Error();const raw=await r.json() as {meta:{scheme_code:number;scheme_name:string;scheme_category:string;fund_house:string};data:{date:string;nav:string}[]};if(String(raw.meta?.scheme_code)!==code||!Array.isArray(raw.data))throw new Error();const points=raw.data.slice(0,2600).flatMap(d=>{const m=/^(\d{2})-(\d{2})-(\d{4})$/.exec(d.date);const price=Number(d.nav);return m&&Number.isFinite(price)&&price>0?[{date:`${m[3]}-${m[2]}-${m[1]}`,price}]:[];}).reverse();const body={name:raw.meta.scheme_name,category:raw.meta.scheme_category,house:raw.meta.fund_house,points,source:'MFapi.in · AMFI-derived NAV history'};cache.set(code,{time:Date.now(),body});if(cache.size>100)cache.delete(cache.keys().next().value!);return Response.json(body,{headers});}catch{return Response.json({error:'Fund history is unavailable. Latest NAV refresh remains separate.'},{status:502,headers});}
}
