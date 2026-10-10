import {getChatGPTUser} from '../../chatgpt-auth';
import {parseNews,type Article} from '../../../lib/market';
export const dynamic='force-dynamic';
const cache=new Map<string,{time:number;articles:Article[]}>();
export async function GET(request:Request){
 const headers={'Cache-Control':'no-store'};
 if(!await getChatGPTUser())return Response.json({error:'Sign in to read this private feed.'},{status:401,headers});
 const p=new URL(request.url).searchParams;const q=(p.get('q')||'').trim().slice(0,120);const category=p.get('category')==='fund'?'fund':'market';const key=q||category;
 try{
  const saved=cache.get(key);if(saved&&Date.now()-saved.time<10*60000)return Response.json({articles:saved.articles,cachedAt:new Date(saved.time).toISOString()},{headers});
  const url=q?`https://news.google.com/rss/search?q=${encodeURIComponent(q+' India finance when:14d')}&hl=en-IN&gl=IN&ceid=IN:en`:category==='fund'?'https://economictimes.indiatimes.com/mf/mf-news/rssfeeds/1107225967.cms':'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms';
  const r=await fetch(url,{signal:AbortSignal.timeout(12000)});if(!r.ok)throw new Error('Feed unavailable.');const articles=parseNews(await r.text(),q?'Google News':'The Economic Times');if(!articles.length)throw new Error('No headlines found.');cache.set(key,{time:Date.now(),articles});if(cache.size>100)cache.delete(cache.keys().next().value!);return Response.json({articles,cachedAt:new Date().toISOString()},{headers});
 }catch{return Response.json({error:'News feed is unavailable right now. Try again later.'},{status:502,headers});}
}
