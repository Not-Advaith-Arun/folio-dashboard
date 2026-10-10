import test from 'node:test';
import assert from 'node:assert/strict';
import {dateISO, parseCSV, importRows, guessMapping, totals, capture, parseNav, mergeHoldings, validatePortfolio} from '../lib/portfolio.ts';
test('import → merge → valuation → snapshots → backup keeps unknown costs and catches malformed inputs', () => {
 const csv='name,type,quantity,invested,price,code,date\n"A, Fund",fund,10,500,65,123456,2026-10-09\nExample stock,stock,2,,100,,2026-10-09\n';
 const rows=parseCSV(csv);const h=importRows(rows.slice(1),guessMapping(rows[0]),'stock','2026-10-09');
 assert.equal(h[0].name,'A, Fund');assert.equal(h[1].invested,null);
 const t=totals(h);assert.equal(t.value,850);assert.equal(t.gain,150);assert.equal(t.knownCost,1);assert.equal(t.complete,false);
 const saved=capture({holdings:h,snapshots:[]});assert.equal(saved.snapshots[0].invested,null);
 assert.deepEqual(validatePortfolio(JSON.parse(JSON.stringify(saved))),saved);
 const changed=mergeHoldings(h,[{...h[0],id:'new',quantity:20}]);assert.equal(changed.length,2);assert.equal(changed[0].id,h[0].id);assert.equal(changed[0].quantity,20);
 assert.throws(()=>parseCSV('name\n"unclosed'),/Unclosed/);
 assert.throws(()=>importRows([['Invalid','fund','-1','500','65','123456','2026-10-09']],guessMapping(rows[0]),'fund','2026-10-09'),/Data row 1/);
 assert.equal(capture({holdings:[{...h[0],price:null}],snapshots:[]}).snapshots.length,0);
});
test('AMFI current and legacy report formats preserve exact plan and dated prices', () => {
 const current='Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date\n123456;INF000000001;-;Example Fund;Direct Plan;Growth Option;65.25;09-Oct-2026';
 const nav=parseNav(current)[0];assert.equal(nav.price,65.25);assert.equal(nav.name,'Example Fund · Direct Plan · Growth Option');assert.equal(nav.date,'09-Oct-2026');assert.equal(dateISO(nav.date),'2026-10-09');assert.throws(()=>dateISO('2026-02-31'),/Invalid/);
 const legacy='Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date\n123456;INF000000001;-;Example Fund Direct Growth;62.5;08-Oct-2026';assert.equal(parseNav(legacy)[0].price,62.5);
 assert.throws(()=>parseNav('<html>Error</html>'),/unrecognised/);
});
