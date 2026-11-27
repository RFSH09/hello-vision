"""A self-contained local HTML page for naming real product crops.

No server and no network: open the file in a browser, click a product (or "not a product" /
"unsure") for each item, then "Download labels.json". Progress is kept in the browser.
"""

from __future__ import annotations

import json
from typing import Any

NOT_PRODUCT = "__not_product__"
UNSURE = "__unsure__"


def render(tasks: list[dict[str, Any]], skus: list[str], sheet_id: str) -> str:
    """Return the HTML page.

    Args:
        tasks: Each has ``id``, ``order``, ``video``, ``crops`` (relative paths),
            ``order_items`` (``{sku: qty}``), ``suggested`` (sku or ``None``).
        skus: All SKU ids; each needs a reference image ``ref/<sku>.jpg``.
        sheet_id: Distinguishes saved progress of different sheets.
    """
    data = json.dumps({"tasks": tasks, "skus": skus, "id": sheet_id}).replace("<", "\\u003c")
    return _PAGE.replace("__DATA__", data)


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Label real crops</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#15171a;color:#e8e8e8;font:15px system-ui,sans-serif}
header{position:sticky;top:0;background:#222629;padding:10px 16px;z-index:2;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
button{font:inherit;color:inherit;background:#33383d;border:1px solid #4a5056;border-radius:6px;padding:6px 10px;cursor:pointer}
button:hover{background:#41484f}button.on{background:#1f7a3d;border-color:#35b360}
main{padding:12px 16px;max-width:1200px;margin:auto}
.card{border:1px solid #3a4046;border-radius:10px;padding:12px;margin:0 0 14px}.card.done{border-color:#35b360}
.crops{display:flex;gap:8px;flex-wrap:wrap}.crops img{height:190px;max-width:260px;object-fit:contain;background:#000;border-radius:6px;cursor:pointer;border:3px solid transparent}
.crops img.off{opacity:.25;border-color:#c0392b}
.choices{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:stretch}
.sku{display:flex;flex-direction:column;align-items:center;gap:4px;width:112px;font-size:12px;overflow-wrap:anywhere}
.sku img{width:92px;height:92px;object-fit:contain;background:#fff;border-radius:4px}
.all{display:none;margin-top:8px;border-top:1px dashed #4a5056;padding-top:8px}.all.show{display:flex}
small{color:#aab}
</style></head><body>
<header><b id="count"></b><button id="next">Next unlabelled</button>
<button id="dl">Download labels.json</button>
<button id="imp">Import labels.json</button><input id="file" type="file" accept=".json,application/json" hidden>
<select id="show" aria-label="filter"><option value="all">show all</option><option value="todo">unlabelled</option><option value="__unsure__">unsure</option><option value="__not_product__">not a product</option><option value="product">products</option></select>
<small id="msg"></small>
<small>Click a crop to exclude it (bad / wrong object). Pick "Not a product" for hands, phones, shelf clutter.</small></header>
<main id="main"></main>
<script>
const D=__DATA__, NP="__not_product__", UN="__unsure__", KEY="hv-labels-"+D.id;
let S={labels:{},excluded:{}};
try{S=Object.assign(S,JSON.parse(localStorage.getItem(KEY)||"{}"))}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const el=(t,a={},...kids)=>{const n=document.createElement(t);for(const[k,v]of Object.entries(a))k==="on"?n.onclick=v:n.setAttribute(k,v);n.append(...kids);return n};
const main=document.getElementById("main"), cards=[];
function count(){const n=D.tasks.filter(t=>S.labels[t.id]).length;document.getElementById("count").textContent=`${n} / ${D.tasks.length} labelled`}
function skuButton(task,sku,note){const b=el("button",{class:"sku","data-v":sku,on:()=>choose(task,sku)},el("img",{src:`ref/${sku}.jpg`,loading:"lazy",alt:sku}),el("span",{},sku));if(note)b.append(el("small",{},note));return b}
function choose(task,v){if(S.labels[task.id]===v)delete S.labels[task.id];else S.labels[task.id]=v;save();paint(task);count()}
function paint(task){const c=cards[D.tasks.indexOf(task)];const v=S.labels[task.id];c.classList.toggle("done",!!v);
 c.querySelectorAll("[data-v]").forEach(b=>b.classList.toggle("on",b.dataset.v===v))}
for(const t of D.tasks){
 const items=Object.entries(t.order_items);
 const crops=el("div",{class:"crops"});
 for(const src of t.crops){const i=el("img",{src,loading:"lazy",alt:"crop"});if(S.excluded[src])i.classList.add("off");
  i.onclick=()=>{S.excluded[src]?delete S.excluded[src]:S.excluded[src]=1;i.classList.toggle("off");save()};crops.append(i)}
 const ch=el("div",{class:"choices"});
 for(const[sku,q]of items)ch.append(skuButton(t,sku,`bought x${q}`+(sku===t.suggested?" · suggested":"")));
 ch.append(el("button",{"data-v":NP,on:()=>choose(t,NP)},"Not a product"),el("button",{"data-v":UN,on:()=>choose(t,UN)},"Unsure"));
 const all=el("div",{class:"choices all"});
 ch.append(el("button",{on:()=>{if(!all.childElementCount)for(const s of D.skus)if(!(s in t.order_items))all.append(skuButton(t,s));all.classList.toggle("show");paint(t)}},"Other product…"));
 const card=el("section",{class:"card"},el("div",{},el("b",{},`${t.order}`),el("small",{},`  ${t.video} · item ${t.id.split(":")[1]} · order contains: `+(items.map(([s,q])=>`${s} x${q}`).join(", ")||"unknown"))),crops,ch,all);
 cards.push(card);main.append(card);
 const v=S.labels[t.id];if(v&&v!==NP&&v!==UN&&!(v in t.order_items)){for(const s of D.skus)if(!(s in t.order_items))all.append(skuButton(t,s));all.classList.add("show")}
 paint(t)}
count();
function addOther(t){const c=cards[D.tasks.indexOf(t)],all=c.querySelector(".all"),v=S.labels[t.id];
 if(v&&v!==NP&&v!==UN&&!(v in t.order_items)){if(!all.childElementCount)for(const s of D.skus)if(!(s in t.order_items))all.append(skuButton(t,s));all.classList.add("show")}}
function refresh(){for(const t of D.tasks){addOther(t);paint(t)}
 document.querySelectorAll(".crops img").forEach(i=>i.classList.toggle("off",!!S.excluded[i.getAttribute("src")]));count();filter()}
function filter(){const f=document.getElementById("show").value;D.tasks.forEach((t,i)=>{const v=S.labels[t.id];
 cards[i].hidden=!(f==="all"||(f==="todo"&&!v)||v===f||(f==="product"&&v&&v!==NP&&v!==UN))})}
document.getElementById("show").onchange=filter;
const file=document.getElementById("file"),msg=document.getElementById("msg");
document.getElementById("imp").onclick=()=>{file.value="";file.click()};
file.onchange=async()=>{if(!file.files.length)return;
 try{const data=JSON.parse(await file.files[0].text());
  if(!data||typeof data.labels!=="object")throw new Error("no labels in this file");
  const ids=new Set(D.tasks.map(t=>t.id)),ok=new Set([NP,UN,...D.skus]);
  const good=Object.entries(data.labels).filter(([k,v])=>ids.has(k)&&ok.has(v));
  if(!good.length)throw new Error("none of its items are on this sheet");
  if(Object.keys(S.labels).length&&!confirm(`Replace the labels of ${good.length} items with the file's?`))return;
  for(const[k,v]of good)S.labels[k]=v;
  S.excluded=Object.fromEntries((data.excluded||[]).map(s=>[s,1]));save();refresh();
  const skipped=Object.keys(data.labels).length-good.length;
  msg.textContent=`imported ${good.length} labels`+(skipped?`, skipped ${skipped} unknown`:"")+(data.sheet&&data.sheet!==D.id?" (file is from a different sheet)":"")+(data.labelled_by?` · source: ${data.labelled_by}`:"");
 }catch(e){msg.textContent="could not import: "+e.message}};
document.getElementById("next").onclick=()=>{const i=D.tasks.findIndex(t=>!S.labels[t.id]);if(i>=0)cards[i].scrollIntoView({behavior:"smooth",block:"center"})};
document.getElementById("dl").onclick=()=>{const out={sheet:D.id,labels:S.labels,excluded:Object.keys(S.excluded)};
 const a=el("a",{href:URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"})),download:"labels.json"});document.body.append(a);a.click();a.remove()};
</script></body></html>
"""
