"use client";
import {useEffect,useState,type ReactNode} from 'react';
import GridLayout,{useContainerWidth,verticalCompactor,type Layout} from 'react-grid-layout';
import {GripHorizontal,SlidersHorizontal,RotateCcw,Check,Move,Maximize2} from 'lucide-react';
import {readWorkspace} from './cloud-sync';
import {persistLocal} from '../lib/client-storage';
export type Widget={id:string;label:string;content:ReactNode;x:number;y:number;w:number;h:number;minH?:number};
export function WidgetBoard({widgets,storageKey}:{widgets:Widget[];storageKey:string}){
 const {width,containerRef,mounted}=useContainerWidth();
 const [layouts,setLayouts]=useState<Record<string,Layout>>({}),[editing,setEditing]=useState(false),[ready,setReady]=useState(false),[notice,setNotice]=useState('');
 const mobile=width<680,cols=width<1000?6:12,bucket=cols===12?'wide':'compact';
 const defaults:Layout=widgets.map((w,i)=>({i:w.id,x:cols===12?w.x:0,y:cols===12?w.y:i*8,w:cols===12?w.w:6,h:w.h,minW:cols===12?3:3,minH:w.minH??4,maxW:cols,maxH:30}));
 const saved=layouts[bucket];const layout=defaults.map(d=>{const s=saved?.find(s=>s.i===d.i);return s?{...d,x:Math.min(s.x,cols-s.w),y:s.y,w:Math.min(s.w,cols),h:Math.max(s.h,d.minH!)}:d;});
 useEffect(()=>{try{setLayouts(readWorkspace(storageKey).layouts??{});}catch{}setReady(true);},[storageKey]);
 function persist(next:Record<string,Layout>){setLayouts(next);try{persistLocal(storageKey+'-layouts',JSON.stringify(next));setNotice('');}catch{setNotice('Layout could not be saved on this device.');}}
 function change(next:Layout){if(!ready)return;persist({...layouts,[bucket]:next.map(({i,x,y,w,h})=>({i,x,y,w,h}))});}
 function adjust(id:string,action:'move'|'size'){
  const current=layout.find(w=>w.i===id)!;const ordered=[...layout].sort((a,b)=>a.y-b.y||a.x-b.x);const next=ordered[(ordered.findIndex(w=>w.i===id)+1)%ordered.length];
  const changed=layout.map(w=>action==='size'?(w.i===id?{...w,w:w.w===cols?(cols===12?4:3):cols,x:0}:w):(w.i===id?{...w,x:Math.min(next.x,cols-w.w),y:next.y}:w.i===next.i?{...w,x:Math.min(current.x,cols-w.w),y:current.y}:w));
  change(verticalCompactor.compact(changed,cols));
 }
 return <div ref={containerRef} className={'widget-board'+(editing?' arranging':'')}><div className="board-toolbar"><span>{mobile?'Cards stack on smaller screens.':editing?'Drag a grip or resize from the corner. Keyboard controls are available on each card.':'Workspace'}</span><div>{!mobile&&editing&&<button className="button" onClick={()=>{persist({});setNotice('Default layout restored.');}}><RotateCcw size={14}/>Reset layout</button>}{!mobile&&<button className={'button'+(editing?' primary':'')} aria-pressed={editing} onClick={()=>setEditing(!editing)}>{editing?<Check size={14}/>:<SlidersHorizontal size={14}/>} {editing?'Done':'Arrange cards'}</button>}</div></div>{notice&&<p className="form-note" role="status">{notice}</p>}
 {mounted&&ready&&!mobile?<GridLayout width={width} layout={layout} gridConfig={{cols,rowHeight:32,margin:[16,16],containerPadding:[0,0]}} dragConfig={{enabled:editing,handle:'.widget-grip',cancel:'.widget-adjust'}} resizeConfig={{enabled:editing,handles:['se']}} onDragStop={next=>change(next)} onResizeStop={next=>change(next)}>{widgets.map(w=><div key={w.id} className="widget-shell" data-widget={w.id}><div className="widget-content">{w.content}</div>{editing&&<div className="widget-tools"><button className="widget-grip" aria-label={'Drag '+w.label} title={'Drag '+w.label}><GripHorizontal size={16}/></button><button className="widget-adjust" aria-label={'Move '+w.label+' to next position'} onClick={()=>adjust(w.id,'move')} title="Move to next position"><Move size={13}/></button><button className="widget-adjust" aria-label={'Change '+w.label+' width'} onClick={()=>adjust(w.id,'size')} title="Toggle width"><Maximize2 size={13}/></button></div>}</div>)}</GridLayout>:<div className="widget-stack">{widgets.map(w=><div className="widget-shell" key={w.id} data-widget={w.id}>{w.content}</div>)}</div>}
 </div>;
}
