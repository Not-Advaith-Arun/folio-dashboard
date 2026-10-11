"use client";
import {useState} from 'react';
import {ResponsiveContainer,LineChart,Line,XAxis,YAxis,Tooltip,Legend,CartesianGrid} from 'recharts';
import {Card} from './ui/card';
import type {ResearchReport} from '../lib/research';

type Policy=NonNullable<NonNullable<ResearchReport['audit']>['policy']>;
const pct=(v:number)=>`${(v*100).toFixed(2)}%`;
const colors=['#bdbdbd','#858585','#a5b8c8','#696969'];
export function ResearchPolicy({policy:p}:{policy:Policy}){
 const [capital,setCapital]=useState(100000);
 const money=(v:number)=>new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0}).format(v);
 const values=p.curve.map(r=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,k==='date'?v:Number(v)*capital])));
 return <Card className="panel research-card"><div className="panel-heading"><div><h2>Absolute risk & capital</h2><span className="muted">Separate challenger · {p.version} · {p.curve[0].date} → {p.curve.at(-1)!.date}</span></div><span className="pill">Paper only</span></div>
 <label style={{display:'flex',gap:12,alignItems:'center',padding:'0 20px 12px'}}>Starting paper capital (INR) <input type="number" aria-label="Starting paper capital" min="1" max="1000000000" value={capital} onChange={e=>{const n=e.currentTarget.valueAsNumber;if(Number.isFinite(n)&&n>=1&&n<=1e9)setCapital(n);}}/></label>
 <div className="research-chart"><ResponsiveContainer width="100%" height={240}><LineChart data={values}><CartesianGrid strokeDasharray="3 3" stroke="var(--border)"/><XAxis dataKey="date" minTickGap={65}/><YAxis domain={['auto','auto']} width={70} tickFormatter={v=>`${(Number(v)/1000).toFixed(0)}k`}/><Tooltip formatter={v=>money(Number(v))}/><Legend/>{p.metrics.map((m,i)=><Line key={m.name} dataKey={m.name} stroke={colors[i]} dot={false} strokeWidth={1.7} strokeDasharray={m.name==='Trend-only ablation'?'3 3':m.name==='Original pipeline'?'7 3':undefined}/>)}</LineChart></ResponsiveContainer></div>
 <div className="table-scroll"><table><thead><tr><th>Policy</th><th>Ending balance</th><th>Net return</th><th>Exit-sampled drawdown</th></tr></thead><tbody>{p.metrics.map(m=><tr key={m.name}><td>{m.name}</td><td>{money(capital*(1+m.totalReturn))}</td><td>{pct(m.totalReturn)}</td><td>{pct(m.maxDrawdown)}</td></tr>)}</tbody></table></div>
 <div className="research-metrics"><div><span>Next challenger exposure</span><strong>{pct(p.latest.exposure)}</strong></div><div><span>Cash / eligible breadth</span><strong>{pct(p.latest.cashWeight)} / {pct(p.latest.breadth)}</strong></div><div><span>Active holding windows</span><strong>{p.activeWindows} / {p.periods}</strong></div></div>
 <p className="research-caption">{p.latest.reason}. {p.protocol.rule}</p>
 <p className="research-caption">{p.status} Balances compound actual simulated net returns, not forecast estimates. Trend-only uses the same controls without ML ranking. Cash earns zero; fractional positions and assumed costs exclude taxes and broker execution.</p>
 <details className="research-method"><summary>Next weights, cost stress & evidence</summary><p>{p.latest.allocations.length?p.latest.allocations.map(a=>`${a.symbol}: ${pct(a.weight)}`).join(' · '):'Entire allocation held in cash.'}</p><p>95% mean excess interval versus equal weight: {pct(p.excessMeanInterval.lower)} to {pct(p.excessMeanInterval.upper)}.</p><p>{p.costStress.map(m=>`${m.costBps} bps: ${pct(m.totalReturn)} return`).join(' · ')}</p>{p.references.map(r=><p key={r.url}><a href={r.url} target="_blank" rel="noreferrer">{r.title}</a><br/>{r.adaptation}</p>)}<ul>{p.limitations.map(l=><li key={l}>{l}</li>)}</ul></details>
 </Card>;
}
