import { assetTypes, dateISO, validatePortfolio, type AssetType, type Portfolio } from './portfolio.ts';
export type Workspace = { portfolio: Portfolio; watchlist: string[] | null; flows: {id:string;date:string;amount:number}[]; targets: Record<AssetType,number> | null; theme: 'light' | 'dark' | null };
export function validateWorkspace(input: unknown): Workspace {
 const w = input as Workspace;
 if (!w || typeof w !== 'object') throw new Error('Invalid workspace.');
 const portfolio = validatePortfolio(w.portfolio);
 if (w.watchlist !== null && (!Array.isArray(w.watchlist) || w.watchlist.length > 50 || w.watchlist.some(s => typeof s !== 'string' || !/^[A-Z0-9&._-]{1,30}\.(NS|BO)$/.test(s)))) throw new Error('Invalid watchlist.');
 if (!Array.isArray(w.flows) || w.flows.length > 1000 || w.flows.some(f => !f || typeof f.id !== 'string' || !f.id || f.id.length > 100 || !Number.isFinite(f.amount) || Math.abs(f.amount) > 1e12)) throw new Error('Invalid cash flows.');
 const flows = w.flows.map(f => ({id:f.id,date:dateISO(f.date),amount:f.amount}));
 if (new Set(flows.map(f=>f.id)).size !== flows.length) throw new Error('Duplicate cash flows.');
 if (w.targets !== null && (!w.targets || assetTypes.some(t => !Number.isFinite(w.targets![t]) || w.targets![t] < 0 || w.targets![t] > 100))) throw new Error('Invalid allocation targets.');
 if (w.theme !== null && w.theme !== 'light' && w.theme !== 'dark') throw new Error('Invalid theme.');
 return {portfolio,watchlist:w.watchlist === null ? null : [...new Set(w.watchlist)],flows,targets:w.targets===null?null:Object.fromEntries(assetTypes.map(t=>[t,w.targets![t]])) as Record<AssetType,number>,theme:w.theme};
}
