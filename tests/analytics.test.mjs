import test from 'node:test';
import assert from 'node:assert/strict';
import {xirr,project} from '../lib/analytics.ts';
// Independent oracle: Microsoft's published XIRR example.
test('XIRR agrees with Microsoft example and rejects incomplete or ambiguous flows',()=>{
 const flows=[['2008-01-01',-10000],['2008-03-01',2750],['2008-10-30',4250],['2009-02-15',3250],['2009-04-01',2750]].map(([date,amount])=>({date,amount}));
 assert.ok(Math.abs(xirr(flows)-.373362535)<1e-8);
 assert.equal(xirr([{date:'2026-01-01',amount:100}]),null);
 assert.equal(xirr([{date:'2025-01-01',amount:-100},{date:'2026-01-01',amount:110}]).toFixed(8),'0.10000000');
 assert.equal(xirr([{date:'2023-01-01',amount:-100},{date:'2024-01-01',amount:230},{date:'2024-12-31',amount:-132}]),null);
 const zero=project(5000,1000,0,2).at(-1);assert.equal(zero.value,29000);assert.equal(zero.invested,29000);
});
