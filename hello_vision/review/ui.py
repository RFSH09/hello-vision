"""Server-rendered HTML for the review UI (all dynamic text is escaped)."""

from __future__ import annotations

import html
import json
from typing import Any

_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:70rem;margin:1.5rem auto;padding:0 1rem}"
    "img{max-width:100%;border:1px solid #999}table{border-collapse:collapse}"
    "td,th{border:1px solid #bbb;padding:.25rem .6rem;text-align:left}"
    ".warn{color:#a40000;font-weight:600}"
)


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style>{body}"
    )


def list_page(rows: list[dict[str, Any]]) -> str:
    """Overview of stored sessions; sessions needing review are highlighted."""
    lines = ["<h1>Sessions</h1><table><tr><th>session</th><th>review</th><th>cart</th></tr>"]
    for r in rows:
        cart = ", ".join(f"{html.escape(i['sku'])} x{int(i['qty'])}" for i in r["items"]) or "-"
        flag = "<span class=warn>needs review</span>" if r["needs_review"] else "ok"
        sid = html.escape(r["session_id"], quote=True)
        lines.append(
            f"<tr><td><a href='/review/{sid}'>{sid}</a></td><td>{flag}</td><td>{cart}</td></tr>"
        )
    lines.append("</table>")
    return _page("Sessions", "".join(lines))


def session_page(summary: dict[str, Any], correction: dict[str, Any] | None) -> str:
    """Replay page: frame slider over the evidence clip, cart, reasons, events, correction form."""
    sid = html.escape(summary["session_id"], quote=True)
    n = int(summary["num_frames"])
    reasons = "".join(f"<li class=warn>{html.escape(r)}</li>" for r in summary["reasons"])
    events = "".join(
        f"<tr><td>{html.escape(e['action'])}</td><td>{html.escape(str(e['sku']))}</td>"
        f"<td>{e['frames'][0]}-{e['frames'][1]}</td><td>{e['confidence']:.2f}</td></tr>"
        for e in summary["events"]
    )
    cart = json.dumps([{"sku": i["sku"], "qty": i["qty"]} for i in summary["items"]])
    prefill = json.dumps((correction or {}).get("items", json.loads(cart)))
    status = "needs review" if summary["needs_review"] else "auto"
    # json.dumps output is placed in a <script>; "<" is escaped so it cannot close the tag
    safe_prefill = prefill.replace("<", "\\u003c")
    body = f"""
<p><a href="/review">&larr; all sessions</a></p>
<h1>Session {sid}</h1>
<p>Status: <b>{status}</b> | confidence {summary["session_confidence"]:.2f} | model {html.escape(summary["model_version"])}</p>
<ul>{reasons}</ul>
<img id=f alt="evidence frame (faces blurred)" src="/review/{sid}/frame/0.jpg">
<p><input id=s type=range min=0 max={max(0, n - 1)} value=0 style="width:100%"> frame <span id=i>0</span> / {n}</p>
<h2>Events</h2><table><tr><th>action</th><th>sku</th><th>frames</th><th>conf</th></tr>{events}</table>
<h2>Correct the cart</h2>
<p>One line per item: <code>sku qty</code>. An empty box means nothing was taken.</p>
<textarea id=c rows=5 cols=40></textarea><br>
<input id=note placeholder="note (optional)"> <button id=b>Save correction</button> <span id=m></span>
<script>
const sid="{sid}", s=document.getElementById('s'), f=document.getElementById('f');
s.oninput=()=>{{document.getElementById('i').textContent=s.value;f.src=`/review/${{sid}}/frame/${{s.value}}.jpg`}};
const pre={safe_prefill};
document.getElementById('c').value=pre.map(x=>x.sku+' '+x.qty).join('\\n');
document.getElementById('b').onclick=async()=>{{
  const items=document.getElementById('c').value.split('\\n').map(l=>l.trim()).filter(Boolean)
    .map(l=>{{const p=l.split(/\\s+/);return {{sku:p[0],qty:parseInt(p[1]||'1')}}}});
  const r=await fetch(`/review/${{sid}}/correction`,{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{items,note:document.getElementById('note').value}})}});
  document.getElementById('m').textContent=r.ok?'saved':'error '+r.status;
}};
</script>"""
    return _page(f"Session {summary['session_id']}", body)


def login_page() -> str:
    """Token login form; the token is exchanged for an HttpOnly cookie, never kept in URLs."""
    body = """
<h1>Review login</h1>
<input id=t type=password placeholder="review token" autocomplete=off>
<button id=b>Log in</button> <span id=m></span>
<script>
document.getElementById('b').onclick=async()=>{
  const r=await fetch('/review/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:document.getElementById('t').value})});
  if(r.ok){location.href='/review'}else{document.getElementById('m').textContent='wrong token'}
};
</script>"""
    return _page("Review login", body)
