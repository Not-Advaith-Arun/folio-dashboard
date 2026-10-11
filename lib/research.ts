import {z} from 'zod';

const number=z.number().finite();
const date=z.string().regex(/^\d{4}-\d{2}-\d{2}$/).refine(v=>Number.isFinite(Date.parse(v)));
const timestamp=z.string().datetime({offset:true});
const symbol=z.string().regex(/^[A-Z0-9&_-]{1,25}\.(NS|BO)$/);
const modelNames=['Elastic Net','Random Forest','Gradient Boosting'] as const;
const models=z.object({'Elastic Net':number,'Random Forest':number,'Gradient Boosting':number});
const forecast=z.object({id:z.string().max(80),createdAt:timestamp,signalDate:date,modelId:z.string().max(50),experiment:z.string().max(50),universe:z.array(symbol).min(4).max(24),pricePicks:z.array(symbol).min(1).max(10),newsPicks:z.array(symbol).min(1).max(10),costBps:number.min(0).max(100),status:z.enum(['pending','resolved']),newsEnabled:z.boolean(),entryDate:date.optional(),exitDate:date.optional(),priceReturn:number.optional(),newsReturn:number.optional(),benchmarkReturn:number.optional()}).superRefine((f,ctx)=>{if(f.status==='resolved'&&(!f.entryDate||!f.exitDate||f.priceReturn===undefined||f.newsReturn===undefined||f.benchmarkReturn===undefined))ctx.addIssue({code:'custom',message:'Resolved forecast is missing outcomes.'});if(f.pricePicks.some(s=>!f.universe.includes(s))||f.newsPicks.some(s=>!f.universe.includes(s)))ctx.addIssue({code:'custom',message:'Forecast picks must belong to the universe.'});});
const reportSchema=z.object({
 version:z.literal(1),engineVersion:z.literal('folio-research-1'),id:z.string().max(80),modelId:z.string().max(50),dataFingerprint:z.string().regex(/^[a-f0-9]{64}$/).optional(),textModelRevision:z.string().regex(/^[a-f0-9]{40}$/).optional(),generatedAt:timestamp,marketAsOf:date,source:z.string().max(250),
 config:z.object({symbols:z.array(symbol).min(4).max(24),names:z.record(z.string().max(160)).optional(),years:z.number().int().min(3).max(5),topK:z.number().int().min(1).max(10),costBps:number.min(0).max(100),includeNews:z.boolean()}),
 coverage:z.object({requested:z.number().int().min(4).max(24),loaded:z.number().int().min(4).max(24),rows:z.number().int().positive().max(100000),trainingEnd:date,newsArticles:z.number().int().min(0).max(300)}),
 rankings:z.array(z.object({symbol,name:z.string().max(160),price:number.positive(),priceScore:number.min(0).max(1),newsScore:number.min(-1).max(1),score:number.min(-.15).max(1.15),headlines:z.number().int().min(0).max(12),modelSpread:number.nonnegative(),models})).min(4).max(24),
 backtest:z.object({folds:z.array(z.object({trainStart:date,trainEnd:date,lastTrainingOutcome:date,testStart:date,testEnd:date,trainingRows:z.number().int().positive(),testRows:z.number().int().positive()})).length(3),curve:z.array(z.object({date,Ensemble:number.positive(),'Equal weight':number.positive(),Momentum:number.positive(),...Object.fromEntries(modelNames.map(n=>[n,number.positive()]))})).min(2).max(200),metrics:z.array(z.object({name:z.enum(['Ensemble','Equal weight','Momentum',...modelNames]),totalReturn:number,cagr:number,maxDrawdown:number.min(-1).max(0),sharpe:number,rankIC:number.min(-1).max(1).nullable()})).length(6),periods:z.number().int().positive().max(100),start:date,end:date}),
 articles:z.array(z.object({id:z.string().max(100),symbol,title:z.string().max(700),url:z.string().url().max(2500).refine(v=>new URL(v).protocol==='https:'),source:z.string().max(200),publishedAt:timestamp,firstSeenAt:timestamp,sentiment:number.min(-1).max(1),label:z.enum(['positive','negative','neutral'])})).max(300),
 forecasts:z.array(forecast).max(100),warnings:z.array(z.string().max(500)).max(100),
}).superRefine((r,ctx)=>{
 if(new Set(r.config.symbols).size!==r.config.symbols.length||r.rankings.length!==r.coverage.loaded||r.rankings.some(s=>!r.config.symbols.includes(s.symbol))||new Set(r.rankings.map(s=>s.symbol)).size!==r.rankings.length)ctx.addIssue({code:'custom',message:'Inconsistent instrument coverage.'});
 if(r.backtest.folds.some(f=>f.lastTrainingOutcome>=f.testStart||f.trainEnd>=f.testStart))ctx.addIssue({code:'custom',message:'Training outcomes overlap evaluation.'});
 if(new Set(r.backtest.metrics.map(m=>m.name)).size!==6)ctx.addIssue({code:'custom',message:'Duplicate performance metrics.'});
});
export type ResearchReport=z.infer<typeof reportSchema>;
export function validateResearchReport(value:unknown):ResearchReport{
 const result=reportSchema.safeParse(value);
 if(!result.success)throw new Error('This is not a valid Folio research report. Your existing report has been kept.');
 return result.data;
}
export const researchSymbols=['RELIANCE.NS','TCS.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','ITC.NS','LT.NS','SBIN.NS','BHARTIARTL.NS','HINDUNILVR.NS','SUNPHARMA.NS','MARUTI.NS'];
