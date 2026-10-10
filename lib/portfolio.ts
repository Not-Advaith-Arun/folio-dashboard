export const assetTypes = ["stock", "fund", "etf", "bond", "cash", "other"] as const;
export type AssetType = typeof assetTypes[number];
export type Holding = { id: string; name: string; type: AssetType; quantity: number; invested: number | null; price: number | null; code?: string; isin?: string; symbol?: string; sector?: string; date: string; source: "import" | "manual" | "amfi" | "market" | "demo" };
export type Snapshot = { date: string; value: number; invested: number | null };
export type Portfolio = { holdings: Holding[]; snapshots: Snapshot[] };
export type Nav = { code: string; isin: string; reinvestmentIsin: string; name: string; price: number; date: string };
export const money = (n: number) => new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);
export const pct = (n: number) => `${n > 0 ? "+" : ""}${n.toFixed(2)}%`;
export function dateISO(value: string): string {
 const plain = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
 const report = /^(\d{1,2})-([A-Za-z]{3})-(\d{4})$/.exec(value);
 let result: string;
 if (plain) result = value;
 else if (report) { const month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"].indexOf(report[2].toLowerCase()); if (month < 0) throw new Error("Invalid price date."); result = `${report[3]}-${String(month+1).padStart(2,"0")}-${report[1].padStart(2,"0")}`; }
 else if (/^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value))) return new Date(value).toISOString().slice(0,10);
 else throw new Error("Use YYYY-MM-DD for price dates.");
 const parsed = new Date(`${result}T12:00:00Z`);
 if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0,10) !== result) throw new Error("Invalid price date.");
 return result;
}
export function totals(holdings: Holding[]) {
 let value = 0, invested = 0, comparableValue = 0, knownCost = 0, unpriced = 0;
 for (const h of holdings) {
  if (h.price === null) { unpriced++; continue; }
  value += h.quantity * h.price;
  if (h.invested !== null) { invested += h.invested; comparableValue += h.quantity * h.price; knownCost++; }
 }
 return { value, invested, gain: comparableValue - invested, gainPct: invested ? (comparableValue - invested) / invested * 100 : null, knownCost, unpriced, complete: !unpriced && knownCost === holdings.length };
}
export function validateHolding(value: unknown): Holding {
 const h = value as Holding;
 if (!h || typeof h !== "object" || typeof h.id !== "string" || !h.id || typeof h.name !== "string" || !h.name.trim() || h.name.length > 250 || !assetTypes.includes(h.type) || !Number.isFinite(h.quantity) || h.quantity <= 0 || h.quantity > 1e12 || (h.price !== null && (!Number.isFinite(h.price) || h.price < 0 || h.price > 1e12)) || (h.invested !== null && (!Number.isFinite(h.invested) || h.invested < 0 || h.invested > 1e15)) || !["import", "manual", "amfi", "market", "demo"].includes(h.source) || typeof h.date !== "string" || !Number.isFinite(Date.parse(h.date)) || (h.code !== undefined && !/^\d{4,8}$/.test(h.code)) || (h.isin !== undefined && !/^[A-Z0-9]{12}$/.test(h.isin))) throw new Error("Invalid holding. Check the name, positive units, non-negative values, date and identifiers.");
 if (h.symbol !== undefined && !/^[A-Z0-9&._-]{1,30}\.(NS|BO)$/.test(h.symbol)) throw new Error("Use an exchange symbol such as RELIANCE.NS or 500325.BO.");
 if (h.sector !== undefined && (typeof h.sector !== "string" || h.sector.length > 150)) throw new Error("Invalid sector label.");
 return { id: h.id, name: h.name.trim(), type: h.type, quantity: h.quantity, invested: h.invested, price: h.price, code: h.code, isin: h.isin, symbol: h.symbol, sector: h.sector, date: dateISO(h.date), source: h.source };
}
export function validatePortfolio(value: unknown): Portfolio {
 const p = value as Portfolio;
 if (!p || !Array.isArray(p.holdings) || p.holdings.length > 2000 || !Array.isArray(p.snapshots) || p.snapshots.length > 2000) throw new Error("This is not a supported Folio backup.");
 const holdings = p.holdings.map(validateHolding);
 if (new Set(holdings.map(h => h.id)).size !== holdings.length) throw new Error("Duplicate holding IDs in backup.");
 const snapshots = p.snapshots.map(s => {
  if (!s || !Number.isFinite(Date.parse(s.date)) || !Number.isFinite(s.value) || s.value < 0 || (s.invested !== null && (!Number.isFinite(s.invested) || s.invested < 0))) throw new Error("Invalid snapshot in backup.");
  return { date: s.date, value: s.value, invested: s.invested };
 });
 return { holdings, snapshots };
}
export function capture(p: Portfolio, date = new Date().toISOString()): Portfolio {
 const t = totals(p.holdings);
 if (!p.holdings.length || t.unpriced) return p;
 return { ...p, snapshots: [...p.snapshots, { date, value: t.value, invested: t.complete ? t.invested : null }].slice(-2000) };
}
export function parseCSV(text: string): string[][] {
 const rows: string[][] = []; let row: string[] = [], cell = "", quoted = false;
 const source = text.replace(/^\uFEFF/, "");
 for (let i = 0; i < source.length; i++) {
  const c = source[i];
  if (c === '"') { if (quoted && source[i + 1] === '"') { cell += '"'; i++; } else if (!quoted && cell.trim()) throw new Error("Malformed CSV quotation."); else quoted = !quoted; }
  else if (c === "," && !quoted) { row.push(cell.trim()); cell = ""; }
  else if ((c === "\n" || c === "\r") && !quoted) { if (c === "\r" && source[i + 1] === "\n") i++; row.push(cell.trim()); if (row.some(Boolean)) rows.push(row); row = []; cell = ""; }
  else cell += c;
 }
 if (quoted) throw new Error("Unclosed quotation in CSV.");
 row.push(cell.trim()); if (row.some(Boolean)) rows.push(row);
 return rows;
}
export function numeric(value: string, optional = false): number | null {
 const clean = value.replace(/[₹,\s]/g, "");
 if (!clean && optional) return null;
 if (!clean || !/^\d+(\.\d+)?$/.test(clean)) throw new Error(`Invalid amount: ${value || "blank"}`);
 const n = Number(clean); if (!Number.isFinite(n)) throw new Error("Invalid numeric amount."); return n;
}
export const fields = ["name", "quantity", "invested", "price", "code", "isin", "type", "date", "symbol", "value", "sector"] as const;
export type Field = typeof fields[number];
export type Mapping = Record<Field, number>;
export const aliases: Record<Field, string[]> = {
 name: ["name", "stockname", "companyname", "schemename", "fundname", "scripname", "securityname", "tradingsymbol"],
 quantity: ["quantity", "units", "noofunits", "closingunits", "balanceunits", "netquantity", "qty"],
 invested: ["invested", "investedamount", "investmentamount", "costvalue", "buyvalue", "purchasevalue", "amountinvested", "totalinvestment"],
 price: ["price", "nav", "currentnav", "closingprice", "ltp", "marketprice", "currentprice"],
 symbol: ["symbol", "ticker", "nsecode", "tradingsymbol"], value: ["marketvalue", "currentvalue", "closingvalue", "valuation", "totalvalue"], sector: ["sector", "industry"],
 code: ["code", "schemecode", "amficode"], isin: ["isin", "isincode"], type: ["type", "assettype"], date: ["date", "asof", "navdate", "valuationdate"]
};
export const normalize = (v: string) => v.toLowerCase().replace(/[^a-z0-9]/g, "");
export function guessMapping(row: string[]): Mapping {
 return Object.fromEntries(fields.map(f => [f, row.findIndex(v => aliases[f].includes(normalize(v)))])) as Mapping;
}
export function importRows(rows: string[][], mapping: Mapping, type: AssetType, date: string): Holding[] {
 if (mapping.name < 0 || mapping.quantity < 0) throw new Error("Map the name and units columns first.");
 if (rows.length > 2000) throw new Error("Import up to 2,000 holdings at a time.");
 return rows.filter(r => r.some(Boolean)).map((r, i) => {
  const get = (f: Field) => mapping[f] >= 0 ? (r[mapping[f]] || "").trim() : "";
  try {
   let rowType: AssetType = type;
   if (get("type")) { const t = normalize(get("type")); if (["fund", "mutualfund", "mf"].includes(t)) rowType = "fund"; else if (["stock", "stocks", "equity"].includes(t)) rowType = "stock"; else if (t === "etf") rowType = "etf"; else if (t === "bond" || t === "bonds") rowType = "bond"; else if (t === "cash") rowType = "cash"; else if (t === "other") rowType = "other"; else throw new Error("Unknown asset type."); }
   const quantity = numeric(get("quantity"))!; const unitPrice = numeric(get("price"), true); const marketValue = numeric(get("value"), true);
   return validateHolding({ id: crypto.randomUUID(), name: get("name"), type: rowType, quantity, invested: numeric(get("invested"), true), price: unitPrice ?? (marketValue !== null && quantity > 0 ? marketValue / quantity : null), symbol: get("symbol") ? (/\.(NS|BO)$/.test(get("symbol")) ? get("symbol") : get("symbol") + ".NS") : undefined, sector: get("sector") || undefined, code: get("code") || undefined, isin: get("isin").toUpperCase() || undefined, date: get("date") || date, source: "import" });
  } catch (error) { throw new Error(`Data row ${i + 1}: ${(error as Error).message}`); }
 });
}
export function parseNav(text: string): Nav[] {
 const rows = text.split(/\r?\n/).map(r => r.split(";").map(c => c.trim()));
 const header = rows.find(r => r[0] === "Scheme Code");
 if (!header) throw new Error("AMFI returned an unrecognised report format.");
 const nameIndex = header.indexOf("Scheme Name"), priceIndex = header.indexOf("Net Asset Value"), dateIndex = header.indexOf("Date");
 if ([nameIndex, priceIndex, dateIndex].some(i => i < 0)) throw new Error("Required NAV columns are missing.");
 return rows.filter(r => /^\d+$/.test(r[0]) && Number.isFinite(Number(r[priceIndex])) && Number(r[priceIndex]) > 0).map(r => ({ code: r[0], isin: r[1], reinvestmentIsin: r[2], name: [r[nameIndex], ...(priceIndex > 4 ? r.slice(nameIndex + 1, priceIndex) : [])].filter(Boolean).join(" · "), price: Number(r[priceIndex]), date: r[dateIndex] }));
}
export function mergeHoldings(existing: Holding[], incoming: Holding[]): Holding[] {
 const next = [...existing];
 for (const h of incoming) {
  const matches = next.map((old, i) => ({ old, i })).filter(({old}) => old.type === h.type && ((h.code && old.code === h.code) || (h.isin && old.isin === h.isin) || normalize(old.name) === normalize(h.name)));
  if (matches.length > 1) throw new Error("Multiple existing holdings match this investment. Resolve duplicate entries before updating.");
  if (matches.length) { const { old, i } = matches[0]; next[i] = { ...h, id: old.id, code: h.code || old.code, isin: h.isin || old.isin }; }
  else next.push(h);
 }
 if (next.length > 2000) throw new Error("A portfolio can contain up to 2,000 holdings.");
 return next;
}
export const demo: Portfolio = {
 holdings: [
  { id: "d1", name: "Parag Parikh Flexi Cap · Direct Growth", type: "fund", quantity: 2150, invested: 143000, price: 87.6, code: "122639", date: "2026-10-09", source: "demo" },
  { id: "d2", name: "UTI Nifty 50 Index · Direct Growth", type: "fund", quantity: 760, invested: 108000, price: 163.8, code: "120716", date: "2026-10-09", source: "demo" },
  { id: "d3", name: "HDFC Bank", type: "stock", quantity: 90, invested: 139500, price: 1742.2, date: "2026-10-09", source: "demo" },
  { id: "d4", name: "Infosys", type: "stock", quantity: 65, invested: 104000, price: 1498.6, date: "2026-10-09", source: "demo" },
  { id: "d5", name: "Nippon India Gold BeES", type: "etf", quantity: 820, invested: 49200, price: 82.4, date: "2026-10-09", source: "demo" },
  { id: "d6", name: "ICICI Prudential Liquid · Direct Growth", type: "fund", quantity: 165, invested: 58000, price: 392.4, date: "2026-10-09", source: "demo" },
  { id: "d7", name: "Tata Consultancy Services", type: "stock", quantity: 24, invested: 88800, price: 3048.5, date: "2026-10-09", source: "demo" },
 ],
 snapshots: [605000, 621500, 614000, 633000, 645000, 638000, 659000, 678000, 667000, 691000, 712000, 700000, 731000, 746000, 762000, 753000, 772513].map((value, i) => ({ date: new Date(Date.UTC(2026, 5, 19 + i * 7)).toISOString(), value, invested: 690500 }))
};
