import {getChatGPTUser} from '../../chatgpt-auth';
import {directory,quote,type Quote} from '../../../lib/market';
export const dynamic='force-dynamic';
const reply=(body:unknown,status=200)=>Response.json(body,{status,headers:{'Cache-Control':'no-store'}});
export async function GET(request:Request){
 if(!await getChatGPTUser())return reply({error:'Sign in to use this private app.'},401);
 const params=new URL(request.url).searchParams;
 try{
  if(params.has('isins')){const ids=(params.get('isins')||'').split(',');if(ids.length>200||ids.some(id=>!/^IN[A-Z0-9]{10}$/.test(id)))return reply({error:'Send up to 200 valid ISINs.'},400);const all=await directory();return reply({securities:all.filter(s=>ids.includes(s.isin))});}
  if(params.get('list')==='nifty'){const all=await directory();return reply({securities:all.filter(s=>s.sector!=='Unclassified'),source:'NSE Nifty 50 constituent directory'});}
  if(params.has('q')||params.has('isin')){const q=(params.get('q')||'').toLowerCase().slice(0,100);const isin=params.get('isin');if(!isin&&q.length<2)return reply({securities:[]});const all=await directory();return reply({securities:all.filter(s=>isin?s.isin===isin:`${s.name} ${s.symbol}`.toLowerCase().includes(q)).slice(0,30)});}
  const symbols=(params.get('symbols')||'').split(',').filter(Boolean);
  if(!symbols.length||symbols.length>12||symbols.some(s=>!/^(?:[A-Z0-9&._-]{1,30}\.(?:NS|BO)|\^NSEI|\^NSEBANK|\^BSESN)$/.test(s)))return reply({error:'Choose up to 12 valid NSE/BSE symbols.'},400);
  const quotes:Quote[]=[];const errors:{symbol:string;message:string}[]=[];
  for(let i=0;i<symbols.length;i+=3){const result=await Promise.allSettled(symbols.slice(i,i+3).map(quote));result.forEach((r,j)=>{if(r.status==='fulfilled')quotes.push(r.value);else errors.push({symbol:symbols[i+j],message:r.reason?.message||'Price unavailable.'});});}
  return reply({quotes,errors,fetchedAt:new Date().toISOString(),source:'Yahoo Finance public chart feed · indicative, may be delayed'});
 }catch{return reply({error:'Market source is unavailable. Saved portfolio prices are unchanged.'},502);}
}
