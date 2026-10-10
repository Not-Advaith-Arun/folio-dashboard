import {parseCSV} from './portfolio.ts';
export type Security = {symbol:string;name:string;isin:string;sector:string};
export type Quote = {symbol:string;name:string;price:number;previousClose:number|null;change:number|null;date:string;currency:string;volume:number|null;high52:number|null;low52:number|null;points:{date:string;price:number}[]};
export type Article = {title:string;url:string;date:string;publisher:string};
const directoryCache:{time:number;data:Security[]}={time:0,data:[]};
const quoteCache=new Map<string,{time:number;data:Quote}>();
export function parseDirectory(csv:string,nifty:string):Security[]{
 const industry=new Map(parseCSV(nifty).slice(1).map(r=>[r[2],r[1]]));
 return parseCSV(csv).slice(1).filter(r=>r[0]&&r[6]).map(r=>({symbol:r[0]+'.NS',name:r[1],isin:r[6],sector:industry.get(r[0])||'Unclassified'}));
}
export async function directory(){
 if(directoryCache.data.length&&Date.now()-directoryCache.time<86400000)return directoryCache.data;
 const responses=await Promise.all([fetch('https://archives.nseindia.com/content/equities/EQUITY_L.csv',{signal:AbortSignal.timeout(12000)}),fetch('https://archives.nseindia.com/content/indices/ind_nifty50list.csv',{signal:AbortSignal.timeout(12000)})]);
 if(!responses[0].ok)throw new Error('NSE security directory unavailable. Enter the exact exchange symbol manually.');
 const [csv,nifty]=await Promise.all([responses[0].text(),responses[1].ok?responses[1].text():Promise.resolve('')]);
 const parsed=parseDirectory(csv,nifty);if(!parsed.length)throw new Error('NSE directory format changed.');
 directoryCache.time=Date.now();directoryCache.data=parsed;return parsed;
}
export function parseQuote(symbol:string,body:unknown):Quote{
 const b=body as {chart?:{result?:{meta:Record<string,unknown>;timestamp?:number[];indicators?:{quote?:{close:(number|null)[]}[]}}[]}};
 const r=b.chart?.result?.[0];if(!r)throw new Error('No quote returned.');const m=r.meta;
 if(m.currency!=='INR'||typeof m.regularMarketPrice!=='number'||!Number.isFinite(m.regularMarketPrice)||m.regularMarketPrice<=0||typeof m.regularMarketTime!=='number')throw new Error('A valid INR quote is unavailable.');
 const num=(v:unknown)=>typeof v==='number'&&Number.isFinite(v)?v:null;
 const points=(r.timestamp||[]).flatMap((time,i)=>{const price=r.indicators?.quote?.[0]?.close[i];return typeof price==='number'&&Number.isFinite(price)&&price>0?[{date:new Date(time*1000).toISOString(),price}]:[];});
 const previousPoint = points.length > 1 ? points[points.length-2].price : null; const prev=previousPoint;const reportedChange=num(m.regularMarketChangePercent)??num(m.fulldayChangePercent);const actualPrev=reportedChange!==null?Number(m.regularMarketPrice)/(1+reportedChange/100):prev;
 return {symbol,name:String(m.longName||m.shortName||symbol),price:Number(m.regularMarketPrice),previousClose:actualPrev,change:reportedChange ?? (prev ? (Number(m.regularMarketPrice) / prev - 1) * 100 : null),date:new Date(Number(m.regularMarketTime)*1000).toISOString(),currency:'INR',volume:num(m.regularMarketVolume),high52:num(m.fiftyTwoWeekHigh),low52:num(m.fiftyTwoWeekLow),points};
}
export async function quote(symbol:string){
 const saved=quoteCache.get(symbol);if(saved&&Date.now()-saved.time<10*60000)return saved.data;
 const response=await fetch(`https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=1y`,{headers:{'User-Agent':'FolioPortfolio/1.0 (personal dashboard)'},signal:AbortSignal.timeout(12000)});
 if(!response.ok)throw new Error(response.status===429?'Price source rate-limited. Try again later.':'Price source unavailable.');
 const parsed=parseQuote(symbol,await response.json());quoteCache.set(symbol,{time:Date.now(),data:parsed});if(quoteCache.size>500)quoteCache.delete(quoteCache.keys().next().value!);return parsed;
}
function decode(v:string){return v.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g,'$1').replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&apos;/g,"'").replace(/&#(\d+);/g,(_,n)=>{const c=Number(n);return c>0&&c<=0x10ffff?String.fromCodePoint(c):'';}).replace(/<[^>]*>/g,'').trim();}
export function parseNews(xml:string,publisher:string):Article[]{
 const tag=(item:string,name:string)=>decode(new RegExp(`<${name}[^>]*>([\\s\\S]*?)</${name}>`,'i').exec(item)?.[1]||'');
 const seen=new Set<string>();
 return [...xml.matchAll(/<item[\s>][\s\S]*?<\/item>/gi)].flatMap(m=>{const item=m[0],url=tag(item,'link'),title=tag(item,'title'),date=tag(item,'pubDate');try{const u=new URL(url);if(u.protocol!=='https:'||!['news.google.com','economictimes.indiatimes.com'].includes(u.hostname)||!title||seen.has(url))return [];seen.add(url);return [{title:title.slice(0,350),url,date:Number.isFinite(Date.parse(date))?new Date(date).toISOString():'',publisher:tag(item,'source')||publisher}];}catch{return [];}}).slice(0,30);
}
