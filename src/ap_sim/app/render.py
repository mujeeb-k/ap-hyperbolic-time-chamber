"""HTML renderer, presentation profile ap-workbench/2 (catalog.PRESENTATION_PROFILE). Formats catalog-typed values
for display; never calculates a value.

Visual reference: a conventional AP operations dashboard (light theme tokens, sidebar, cards,
labels, badges). Rules: no text beyond mapped content; DOM order equals observation order (CSS grid
moves things only visually); accessible names equal visible text; the only data attribute is
data-agent-id; no JavaScript, no external requests.
"""
from __future__ import annotations

from decimal import Decimal
from html import escape

from ap_sim.app import catalog
from ap_sim.app.viewmodel import ControlVM, PageVM, RegionVM
from ap_sim.domain.records import Money, Qty
from ap_sim.events.clock import format_time

NULL = "—"
NUMERIC = ("qty", "money", "int", "time")

# Case-action groups, separated visually by a divider (no captions).
ACTION_GROUPS = (("request_confirmation",), ("quantity", "request_receipt_posting", "post_receipt"),
                 ("reprocess", "release"), ("keep_blocked_shortfall", "keep_blocked_pending", "keep_blocked_buyer_review"))

# Badge tone per enum code (status color only for status meaning).
_GOOD = {"pass", "eligible", "posted_released", "posted", "answered", "accepted"}
_BAD = {"fail", "not_eligible", "rejected", "cancelled", "closed_po"}
_WARN = {"posted_blocked", "flag", "requested"}

STYLE = """
:root{--bg-base:#f7f7f5;--bg-surface:#ffffff;--bg-elevated:#f0f0ed;--bg-active:#e8e8e5;
--border-default:#e2e2de;--border-active:#b0b0aa;--border-subtle:#ededea;
--text-primary:#111110;--text-secondary:#3b3b38;--text-label:#57574f;--neutral:#44443f;--neutral-bg:#ebebe7;
--critical:#991b1b;--critical-bg:#fee2e2;--high:#92400e;--high-bg:#fef3c7;--low:#166534;--low-bg:#dcfce7;
--low-border:#86efac;--link:#2563eb;--field-border:#767672;--focus:#1d4ed8;
--font-mono:"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
--font-sans:-apple-system,"Inter","Segoe UI",system-ui,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body.app{display:grid;grid-template-columns:248px minmax(0,1fr);grid-template-rows:auto 1fr;
grid-template-areas:"side head" "side main";background:var(--bg-base);color:var(--text-primary);
font-family:var(--font-sans);font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased}
.banner{grid-area:head;display:grid;grid-template-columns:1fr auto;grid-template-areas:"title chips" "case advance";
column-gap:16px;row-gap:4px;align-items:center;padding:18px 28px 8px}
.banner h1{grid-area:title;margin:0;font-size:22px;font-weight:600;letter-spacing:-0.01em}
.case{grid-area:case;display:flex;gap:18px;list-style:none;margin:0;padding:0;flex-wrap:wrap;
font-family:var(--font-mono);font-size:13px;color:var(--text-secondary)}
.chips{grid-area:chips;display:flex;gap:8px;list-style:none;margin:0;padding:0;flex-wrap:wrap;justify-content:flex-end}
.chips li{font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);background:var(--bg-surface);
border:1px solid var(--border-default);border-radius:2px;padding:4px 8px}
.manual-advance{grid-area:advance;display:flex;gap:8px;align-items:center;justify-content:flex-end;margin:0;
font-size:13px;color:var(--text-primary)}
.manual-advance select{padding:6px 8px;border:2px solid var(--field-border);border-radius:2px;background:#ffffff;
font-family:var(--font-mono);font-size:13px;color:var(--text-primary)}
.manual-advance button{min-height:34px;padding:6px 12px;border:1px solid var(--field-border);border-radius:2px;
background:var(--bg-surface);color:var(--text-primary);font-size:13px;font-weight:600;cursor:pointer}
select:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.content{grid-area:main;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;align-content:start;
align-items:start;padding:12px 28px 28px}
.card{grid-column:1;background:var(--bg-surface);border:1px solid var(--border-default);border-radius:2px;overflow:hidden}
.card h2,.card-nav h2{margin:0;padding:10px 16px;border-bottom:1px solid var(--border-default);font-family:var(--font-mono);
font-size:12px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-label)}
.card-result,.card-changed{order:-1}
.card-result.ok{border-color:var(--low-border)}
.card-result.ok h2{background:var(--low-bg);color:var(--low)}
.card-result.rejected{border-color:var(--critical)}
.card-result.rejected h2{background:var(--critical-bg);color:var(--critical)}
.card-changed{border-left:3px solid var(--link)}
.card-result td:last-child{white-space:normal;min-width:280px}
.card-actions{grid-column:2;grid-row:1 / span 12;position:sticky;top:12px}
.card-actions form{margin:0;padding:0 12px}
.card-actions .posting{padding-top:2px}
.card-actions form:first-of-type{padding-top:12px}
.btn{display:flex;align-items:center;width:100%;min-height:38px;margin:0 0 6px;padding:6px 12px;border-radius:2px;
border:1px solid var(--field-border);background:var(--bg-surface);color:var(--text-primary);font-family:var(--font-sans);
font-size:13px;font-weight:500;text-align:left;cursor:pointer}
.btn:hover{background:var(--bg-elevated);color:var(--text-primary)}
.field{display:block;margin:0 0 8px;font-family:var(--font-sans);font-size:13px;font-weight:600;color:var(--text-primary)}
.field input{display:block;width:100%;margin-top:4px;padding:8px 10px;border:2px solid var(--field-border);border-radius:2px;
background:#ffffff;font-family:var(--font-mono);font-size:14px;color:var(--text-primary);box-shadow:inset 0 1px 2px rgba(0,0,0,0.08)}
a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.divider{border:0;border-top:1px solid var(--border-default);margin:6px 0 10px}
.table{width:100%;border-collapse:collapse}
.table th{padding:9px 12px;text-align:left;font-family:var(--font-mono);font-size:11px;font-weight:600;letter-spacing:0.06em;
text-transform:uppercase;color:var(--text-label);background:var(--bg-base);border-bottom:1px solid var(--border-default);white-space:nowrap}
.table td{padding:8px 12px;border-bottom:1px solid var(--border-subtle);font-size:13px;color:var(--text-secondary);white-space:nowrap}
.table tbody tr:last-child td{border-bottom:0}
.table tbody tr:hover td{background:var(--bg-elevated)}
.table .num{text-align:right;font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.table td:first-child{font-family:var(--font-mono);color:var(--text-primary)}
.table .empty{text-align:center;color:var(--text-label);padding:14px 12px}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:2px;font-family:var(--font-mono);font-size:12px;
font-weight:600;border:1px solid var(--neutral-bg);background:var(--neutral-bg);color:var(--neutral)}
""" + "".join(
    f".b-{code}{{background:var(--{tone}-bg);color:var(--{tone});border-color:var(--{tone}-bg)}}"
    for tone, codes in (("low", _GOOD), ("critical", _BAD), ("high", _WARN)) for code in sorted(codes)) + """
.card-nav{grid-area:side;background:var(--bg-surface);border-right:1px solid var(--border-default);padding:8px 0}
.card-nav h2{border-bottom:0;padding:16px 18px 8px}
.card-nav ul{list-style:none;margin:0;padding:0 6px}
.nav-item{display:flex;align-items:center;padding:9px 12px;margin:1px 0;border-radius:2px;border:1px solid transparent;
font-size:14px;color:var(--text-secondary);text-decoration:none}
.nav-item:hover{background:var(--bg-elevated);color:var(--text-primary)}
.nav-item.active{background:var(--bg-active);border-color:var(--border-active);color:var(--text-primary);font-weight:500}
"""


def _decimal_text(v: Decimal) -> str:
    return format(v.normalize(), "f")


def format_value(col: catalog.Column, value) -> str:
    if value is None:
        return NULL
    kind = col.type
    if kind == "qty":
        assert isinstance(value, Qty)
        return f"{_decimal_text(value.value)} {value.unit}"
    if kind == "money":
        assert isinstance(value, Money) and value.currency == "EUR"
        return f"€{value.value:,.2f}"
    if kind == "time":
        return format_time(value)
    if kind.startswith("enum:"):
        return catalog.ENUMS[kind[5:]][value]
    return str(value)


def _cell(col: catalog.Column, value) -> str:
    text = escape(format_value(col, value))
    if col.type.startswith("enum:") and value is not None:
        return f'<td><span class="badge b-{escape(value)}">{text}</span></td>'
    return f'<td class="num">{text}</td>' if col.type in NUMERIC else f"<td>{text}</td>"


def _control(c: ControlVM, ids: dict[str, str], current_page: str) -> str:
    cid = escape(ids[c.key])
    name = escape(c.name)
    kind, key = c.key.split(":", 1)
    if c.role == "link":
        active = key == current_page
        return (f'<li><a class="nav-item{" active" if active else ""}"{" aria-current=\"page\"" if active else ""} '
                f'href="/go/{key}" data-agent-id="{cid}">{name}</a></li>')
    if c.role == "textbox":
        return (f'<label class="field">{name} '
                f'<input type="text" name="quantity" data-agent-id="{cid}" value="{escape(c.value or "")}"></label>')
    if key in catalog.QUANTITY_ACTIONS:  # submitted with the quantity box's current value
        return f'<button class="btn" type="submit" name="op" value="{key}" data-agent-id="{cid}">{name}</button>'
    return (f'<form method="post" action="/op/{key}"><button class="btn" type="submit" '
            f'data-agent-id="{cid}">{name}</button></form>')


def _actions(r: RegionVM, ids: dict[str, str], page: str) -> str:
    by_key = {c.key.split(":", 1)[1]: c for c in r.controls}
    groups = ["".join(_control(by_key[k], ids, page) for k in g) for g in ACTION_GROUPS]
    groups[1] = f'<form class="posting" method="post" action="/op/posting">{groups[1]}</form>'
    assert [k for g in ACTION_GROUPS for k in g] == list(by_key), "action groups must keep catalog order"
    return '<hr class="divider">'.join(groups)


def _region(r: RegionVM, ids: dict[str, str], page: str) -> str:
    title = escape(r.title)
    if r.role == "navigation":
        links = "".join(_control(c, ids, page) for c in r.controls)
        return f'<nav class="card-nav" aria-label="{title}"><h2>{title}</h2><ul>{links}</ul></nav>'
    if r.role == "actions":
        return f'<section class="card card-actions" aria-label="{title}"><h2>{title}</h2>{_actions(r, ids, page)}</section>'
    base = catalog.REGION[r.role.split(":", 1)[1] if r.role.startswith("changed:") else r.role]
    cols = ["Record", "Rev"] + [c.header for c in base.columns]
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in cols)
    if r.records:
        body = "".join(
            "<tr>" + f'<td>{escape(rec.id)}</td><td class="num">{rec.version}</td>'
            + "".join(_cell(c, rec.fields[c.field]) for c in base.columns) + "</tr>"
            for rec in r.records)
    else:
        body = f'<tr><td colspan="{len(cols)}" class="empty">{catalog.EMPTY_MESSAGE}</td></tr>'
    kind = "card"
    if r.role == "result":
        kind += " card-result " + ("ok" if r.records[0].fields["outcome"] == "accepted" else "rejected")
    elif r.role.startswith("changed:"):
        kind += " card-changed"
    return (f'<section class="{kind}" aria-label="{title}"><h2>{title}</h2>'
            f'<table class="table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></section>')


MANUAL_ADVANCE_MINUTES = (5, 15, 30, 60, 120, 240, 480)


def _manual_advance(action: str) -> str:
    """Manual play only (never on agent surfaces): a one-time form to let business time pass."""
    options = "".join(f'<option value="{m}"{" selected" if m == 480 else ""}>{m} min</option>'
                      for m in MANUAL_ADVANCE_MINUTES)
    return (f'<form class="manual-advance" method="post" action="{escape(action)}">'
            f'<label>Advance time by <select name="minutes">{options}</select></label>'
            f'<button type="submit">Advance time</button></form>')


def render_page(vm: PageVM, ids: dict[str, str], manual_advance: str | None = None) -> str:
    case = "".join(f"<li>{label}: {escape(vm.case[key])}</li>" for label, key in catalog.CASE_ITEMS)
    banner = (f'<header class="banner"><h1>{escape(vm.title)}</h1><ul class="case">{case}</ul><ul class="chips">'
              f'<li>Business time: {format_time(vm.clock)}</li><li>State version: {vm.state_version}</li>'
              f'<li>Role: {catalog.ROLE_LABEL[vm.role]}</li></ul>'
              f'{_manual_advance(manual_advance) if manual_advance else ""}</header>')
    title = f"{escape(vm.title)} · {catalog.APP_TITLE}"
    result = next((r for r in vm.regions if r.role == "result"), None)
    if result is not None:  # announced on page load; the same facts are in the result region
        rec = result.records[0].fields
        title = f"{catalog.ENUMS['outcome'][rec['outcome']]}: {escape(catalog.ENUMS['action'][rec['action']])} · {title}"
    main = "".join(_region(r, ids, vm.page) for r in vm.regions if r.role != "navigation")
    nav = "".join(_region(r, ids, vm.page) for r in vm.regions if r.role == "navigation")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>{title}</title><style>{STYLE}</style></head>'
            f'<body class="app">{banner}<main class="content">{main}</main>{nav}</body></html>')
