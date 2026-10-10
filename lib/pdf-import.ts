import * as pdfjs from 'pdfjs-dist';
import workerURL from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import {aliases,normalize} from './portfolio';
pdfjs.GlobalWorkerOptions.workerSrc=workerURL;
type Word={str:string;x:number;y:number;width:number};
export async function readPDF(file:File,password:string){
 const task=pdfjs.getDocument({data:new Uint8Array(await file.arrayBuffer()),password,isEvalSupported:false});
 const pdf=await task.promise;
 try{
  if(pdf.numPages>60)throw new Error('Use a statement with up to 60 pages, or split it into smaller files.');
  const pages:{name:string;rows:string[][]}[]=[];
  for(let n=1;n<=pdf.numPages;n++){
   const page=await pdf.getPage(n);const text=await page.getTextContent();
   const words:Word[]=text.items.flatMap(item=>'str' in item&&item.str.trim()?[{str:item.str.trim(),x:item.transform[4],y:item.transform[5],width:item.width}]:[]);
   const lines:{y:number;words:Word[]}[]=[];
   for(const word of words.sort((a,b)=>b.y-a.y||a.x-b.x)){let line=lines.find(l=>Math.abs(l.y-word.y)<3);if(!line){line={y:word.y,words:[]};lines.push(line);}line.words.push(word);}
   const grouped=lines.map(l=>{const cells:Word[]=[];for(const w of l.words.sort((a,b)=>a.x-b.x)){const last=cells.at(-1);if(last&&w.x-last.x-last.width<15){last.str+=' '+w.str;last.width=w.x+w.width-last.x;}else cells.push({...w});}return cells;});
   const header=grouped.find(r=>r.some(c=>aliases.name.includes(normalize(c.str))||/scheme name|security name|description/i.test(c.str))&&r.some(c=>aliases.quantity.includes(normalize(c.str))||/units|quantity|balance/i.test(c.str)));
   const rows=header?grouped.map(r=>{const cells=Array.from({length:header.length},()=>[] as string[]);for(const w of r){let nearest=0,best=Infinity;header.forEach((c,i)=>{const distance=Math.abs(w.x+w.width/2-(c.x+c.width/2));if(distance<best){best=distance;nearest=i;}});cells[nearest].push(w.str);}return cells.map(c=>c.join(' '));}):grouped.map(r=>r.map(w=>w.str));
   if(rows.length)pages.push({name:`PDF page ${n} · review extraction`,rows});
  }
  if(!pages.length)throw new Error('No text could be extracted. Scanned statements need a CSV/XLSX export or manual entry.');
  return pages;
 }finally{await pdf.destroy();}
}
