import test from 'node:test';
import assert from 'node:assert/strict';
import {parseDirectory,parseQuote,parseNews} from '../lib/market.ts';
test('security resolution, daily prices and news preserve source identity and reject unsafe links',()=>{
 const securities=parseDirectory('SYMBOL,NAME,SERIES,DATE,PAID,LOT,ISIN,FACE\nACME,Acme Ltd,EQ,2020,10,1,INE000000001,10','Company Name,Industry,Symbol,Series,ISIN Code\nAcme Ltd,Technology,ACME,EQ,INE000000001');
 assert.equal(securities[0].isin,'INE000000001');assert.equal(securities[0].symbol,'ACME.NS');assert.equal(securities[0].sector,'Technology');
 const q=parseQuote('ACME.NS',{chart:{result:[{meta:{currency:'INR',regularMarketPrice:110,regularMarketTime:1767225600,chartPreviousClose:50},timestamp:[1767052800,1767139200,1767225600],indicators:{quote:[{close:[90,100,110]}]}}]}});
 assert.equal(q.previousClose,100);assert.ok(Math.abs(q.change-10)<1e-8);assert.equal(q.points.length,3);
 assert.throws(()=>parseQuote('ACME.NS',{chart:{result:[{meta:{currency:'USD',regularMarketPrice:10,regularMarketTime:1767225600}}]}}),/INR/);
 const news=parseNews('<rss><item><title><![CDATA[ACME &amp; markets]]></title><link>https://economictimes.indiatimes.com/example</link><pubDate>Sat, 10 Oct 2026 10:00:00 GMT</pubDate></item><item><title>Unsafe</title><link>javascript:alert(1)</link></item></rss>','ET');
 assert.equal(news.length,1);assert.equal(news[0].title,'ACME & markets');assert.equal(news[0].date,'2026-10-10T10:00:00.000Z');
});
