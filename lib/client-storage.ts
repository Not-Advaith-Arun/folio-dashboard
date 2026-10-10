export function persistLocal(key:string,value:string){
 localStorage.setItem(key,value);
 window.dispatchEvent(new CustomEvent('folio-local-change',{detail:{key}}));
}
