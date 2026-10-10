import { getChatGPTUser } from "../../chatgpt-auth";
import { parseNav, type Nav } from "../../../lib/portfolio";
export const dynamic = "force-dynamic";
let cache: { time: number; schemes: Nav[] } | null = null;
const respond = (data: unknown, status = 200) => Response.json(data, { status, headers: { "Cache-Control": "no-store" } });
async function schemes() {
 if (cache && Date.now() - cache.time < 15 * 60 * 1000) return cache.schemes;
 const response = await fetch("https://portal.amfiindia.com/spages/NAVAll.txt", { signal: AbortSignal.timeout(15000) });
 if (!response.ok) throw new Error("AMFI is unavailable.");
 const parsed = parseNav(await response.text());
 if (!parsed.length) throw new Error("No usable NAVs.");
 cache = { time: Date.now(), schemes: parsed }; return parsed;
}
export async function GET(request: Request) {
 if (!await getChatGPTUser()) return respond({ error: "Sign in to use this private app." }, 401);
 const q = new URL(request.url).searchParams.get("q")?.trim().toLowerCase() || "";
 if (q.length < 3 || q.length > 150) return respond({ schemes: [] });
 try { const tokens = q.split(/\s+/); const all = await schemes(); return respond({ schemes: all.filter(s => tokens.every(t => `${s.name} ${s.code} ${s.isin}`.toLowerCase().includes(t))).slice(0, 40) }); }
 catch { return respond({ error: "Cannot fetch AMFI's NAV report right now. Please try again later." }, 502); }
}
export async function POST(request: Request) {
 if (!await getChatGPTUser()) return respond({ error: "Sign in to use this private app." }, 401);
 const origin = request.headers.get("origin");
 if (!origin || origin !== new URL(request.url).origin) return respond({ error: "Request must come from this app." }, 403);
 if (Number(request.headers.get("content-length") || 0) > 30000) return respond({ error: "Request too large." }, 413);
 try {
  const text = await request.text(); if (text.length > 30000) return respond({ error: "Request too large." }, 413);
  const body = JSON.parse(text);
  if (!Array.isArray(body.ids) || body.ids.length > 1000 || body.ids.some((id: unknown) => typeof id !== "string" || !/^(\d{4,8}|[A-Z0-9]{12})$/.test(id))) return respond({ error: "Send valid AMFI scheme codes or ISINs." }, 400);
  const wanted = new Set(body.ids); const all = await schemes();
  return respond({ schemes: all.filter(s => wanted.has(s.code) || wanted.has(s.isin) || wanted.has(s.reinvestmentIsin)), fetchedAt: new Date().toISOString() });
 } catch { return respond({ error: "NAV refresh failed. Your saved holdings and prices are unchanged." }, 502); }
}
