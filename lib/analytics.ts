export type CashFlow={date:string;amount:number};
export function xirr(flows:CashFlow[]):number|null{
 if(flows.length<2||!flows.some(f=>f.amount<0)||!flows.some(f=>f.amount>0))return null;
 const ordered=[...flows].sort((a,b)=>a.date.localeCompare(b.date));const first=Date.parse(ordered[0].date);if(!Number.isFinite(first))return null;
 const times=ordered.map(f=>({amount:f.amount,years:(Date.parse(f.date)-first)/86400000/365}));if(times.some(f=>!Number.isFinite(f.amount)||!Number.isFinite(f.years))||times.at(-1)!.years<=0)return null;
 const npv=(r:number)=>times.reduce((sum,f)=>sum+f.amount/Math.pow(1+r,f.years),0);
 const grid=[-.9999,-.99,-.9,-.75,-.5,-.25,0,.1,.25,.5,1,2,5,10,100,1000,1000000];const roots:number[]=[];
 for(let i=0;i<grid.length-1;i++){let lo=grid[i],hi=grid[i+1],a=npv(lo),b=npv(hi);if(a===0)roots.push(lo);if(a*b<0){for(let j=0;j<200;j++){const mid=(lo+hi)/2,c=npv(mid);if(Math.abs(c)<1e-8){lo=hi=mid;break;}if(a*c<0){hi=mid;b=c;}else{lo=mid;a=c;}}roots.push((lo+hi)/2);}}
 if(npv(grid.at(-1)!)===0)roots.push(grid.at(-1)!);
 return roots.length===1?roots[0]:null;
}
export function project(initial:number,monthly:number,rate:number,years:number){
 const monthlyRate=Math.pow(1+rate/100,1/12)-1;let value=initial;
 return Array.from({length:years+1},(_,year)=>{if(year>0)for(let month=0;month<12;month++)value=value*(1+monthlyRate)+monthly;return {year,value,invested:initial+monthly*12*year};});
}
