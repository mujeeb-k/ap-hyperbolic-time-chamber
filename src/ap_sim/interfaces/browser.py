"""Browser surface: headless Chromium driven by Playwright.

The producer reads only the page's ARIA snapshot (roles, names, values, states, visible text) and
each control's data-agent-id through a role+name locator scoped to its region. Anything it cannot
map is a ProducerError (an environment bug), never silently skipped.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from ap_sim.app import catalog
from ap_sim.app.server import AppServer
from ap_sim.app.session import AppSession
from ap_sim.events.clock import format_time, parse_time
from ap_sim.interfaces.canonical import ParseError, normalize_text, parse_cell


SETTLE_POLL_S = 0.05     # settle loop (adapted from eagleview-rpa replay/settle.py)
SETTLE_DEADLINE_S = 5.0


class ProducerError(RuntimeError):
    pass


@dataclass
class Node:
    role: str
    name: str | None = None
    attrs: tuple[str, ...] = ()
    value: str | None = None
    children: list["Node"] = field(default_factory=list)


_LINE = re.compile(r'^(?P<role>/?[a-z]+)(?: "(?P<name>(?:[^"\\]|\\.)*)")?(?P<attrs>(?: \[[^\]]+\])*)(?::(?: (?P<value>.*))?)?$')


_YAML_SQ = re.compile(r"^'((?:[^']|'')*)'(:(?: .*)?)?$")


def _scalar(raw: str | None) -> str | None:
    if raw is None:
        return None
    if raw.startswith('"'):
        return json.loads(raw)
    if raw.startswith("'"):
        raise ProducerError(f"unsupported single-quoted scalar: {raw!r}")
    return raw


def parse_aria(snapshot: str) -> list[Node]:
    """Parse Playwright's ARIA snapshot (indentation-structured lines '- role "name" [attrs]: value')."""
    root = Node("root")
    stack: list[tuple[int, Node]] = [(-1, root)]
    for line in snapshot.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if not body.startswith("- "):
            raise ProducerError(f"unexpected snapshot line: {line!r}")
        entry = body[2:]
        quoted = _YAML_SQ.match(entry)
        if quoted:  # YAML single-quotes a whole entry when its key contains ': '
            entry = quoted.group(1).replace("''", "'") + (quoted.group(2) or "")
        m = _LINE.match(entry)
        if not m:
            raise ProducerError(f"unparsed snapshot line: {line!r}")
        name = json.loads(f'"{m.group("name")}"') if m.group("name") is not None else None
        attrs = tuple(a.strip("[]") for a in re.findall(r"\[[^\]]+\]", m.group("attrs") or ""))
        node = Node(m.group("role"), name, attrs, _scalar(m.group("value")))
        while stack[-1][0] >= indent:
            stack.pop()
        stack[-1][1].children.append(node)
        stack.append((indent, node))
    return root.children


def _text_of(node: Node) -> str:
    return normalize_text(node.value if node.value is not None else (node.name or ""))


class BrowserSurface:
    name = "browser"

    def __init__(self, headed: bool = False, video_dir: str | None = None, slow_mo: int = 0,
                 viewport: dict | None = None):
        """Defaults: headless, no recording, Playwright's viewport (every test and run). `headed`, `video_dir`,
        `slow_mo` and `viewport` exist only to let a person watch a recorded episode replayed. They change nothing the
        agent sees."""
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=not headed, slow_mo=slow_mo)
        ctx = {"viewport": viewport} if viewport else {}
        if video_dir:
            ctx |= {"record_video_dir": video_dir, **({"record_video_size": viewport} if viewport else {})}
        self._browser_context = self._browser.new_context(**ctx)
        self.page = self._browser_context.new_page()
        self.server: AppServer | None = None
        self.blocked: list[str] = []
        self.page.route("**/*", self._guard)

    def _guard(self, route) -> None:
        """Network guard (adapted from eagleview-rpa): only the local application's origin is reachable."""
        if self.server is not None and route.request.url.startswith(self.server.url + "/"):
            route.continue_()
        else:
            self.blocked.append(route.request.url)
            route.abort()

    # Lifecycle -------------------------------------------------------------------------------------
    def attach(self, session: AppSession, ids: dict[str, str]) -> None:
        if self.server is None:
            self.server = AppServer(session).start()
        self.server.session = session
        self.session, self.ids = session, ids
        self.page.goto(self.server.url + "/view")

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
        self._browser_context.close()   # finalizes a recorded video
        self._browser.close()
        self._pw.stop()

    # Action execution (validity, stale rule, horizon and limit are checked by the tool layer) -------
    def _loc(self, element_id: str):
        return self.page.locator(f'[data-agent-id="{element_id}"]')

    def click(self, key: str, element_id: str) -> None:
        self._loc(element_id).click()
        self.page.wait_for_load_state("load")

    def fill(self, key: str, element_id: str, text: str) -> None:
        """Playwright's fill (fires the input events), then commit the value read back from the field as
        the application's draft. Enter is not used: it would submit the posting form."""
        loc = self._loc(element_id)
        loc.fill(text)
        typed = loc.input_value()
        self.page.request.post(self.server.url + "/ui/draft", form={"value": typed})
        self.page.goto(self.server.url + "/view")

    def observe(self) -> None:
        self.page.goto(self.server.url + "/ui/observe")

    def _locked(self, fn, *args) -> None:
        with self.server.lock:
            fn(*args)
        self.page.goto(self.server.url + "/view")  # refresh the rendered page; no charge

    def advance(self, minutes: int) -> None:
        self._locked(self.session.advance, minutes)

    def charge(self, minutes: int) -> None:
        self._locked(self.session.charge, minutes)

    # Producer --------------------------------------------------------------------------------------
    def _settled(self) -> tuple[list[Node], dict]:
        """Poll until the banner shows the engine's current state (deadline: infrastructure failure)."""
        eng = self.session.engine
        want = (eng.state_version, format_time(eng.clock))
        deadline = time.monotonic() + SETTLE_DEADLINE_S
        last = "no snapshot"
        while True:
            try:
                nodes = parse_aria(self.page.aria_snapshot())
                banners = [n for n in nodes if n.role == "banner"]
                if len(banners) == 1:
                    context = self._context(banners[0])
                    if (context["state_version"], context["clock"]) == want:
                        return nodes, context
                    last = f"banner shows {context['state_version']}/{context['clock']}, engine {want}"
            except PlaywrightError as exc:  # mid-navigation: look again
                last = str(exc).splitlines()[0]
            if time.monotonic() > deadline:
                raise ProducerError(f"page did not settle: {last}")
            time.sleep(SETTLE_POLL_S)

    def produce(self) -> dict:
        nodes, context = self._settled()
        if self.blocked:
            raise ProducerError(f"blocked non-local requests: {self.blocked}")
        if [n.role for n in nodes] != ["banner", "main", "navigation"]:
            raise ProducerError(f"top-level nodes must be banner, main, navigation: {[n.role for n in nodes]}")
        regions = [self._region(n) for n in nodes[1].children]
        regions.append(self._region(nodes[2], page=context["page"]))
        return {**context, "regions": regions}

    def _context(self, banner: Node) -> dict:
        if banner.name is not None:
            raise ProducerError("the banner is unnamed (presentation profile ap-workbench/1)")
        lists: list[list[tuple[str, str]]] = []
        page = None
        for child in banner.children:
            if child.role == "heading" and "level=1" in child.attrs and page is None:
                page = normalize_text(child.name or "")
                continue
            if child.role != "list":
                raise ProducerError(f"unmapped node in the banner: {child.role}")
            items = []
            for item in child.children:
                if item.role != "listitem" or item.value is None or ": " not in item.value:
                    raise ProducerError("banner items must be 'Label: value' list items")
                label, value = item.value.split(": ", 1)
                items.append((label, normalize_text(value)))
            lists.append(items)
        case_labels = [label for label, _ in catalog.CASE_ITEMS]
        if len(lists) != 2 or [l for l, _ in lists[0]] != case_labels \
                or [l for l, _ in lists[1]] != ["Business time", "State version", "Role"]:
            raise ProducerError(f"unexpected banner lists: {lists}")
        case, items = dict(lists[0]), dict(lists[1])
        try:
            return {"state_version": int(items["State version"]), "clock": format_time(parse_time(items["Business time"])),
                    "role": {v: k for k, v in catalog.ROLE_LABEL.items()}[items["Role"]],
                    "page": catalog.PAGE_BY_TITLE[page],
                    "case": {key: case[label] for label, key in catalog.CASE_ITEMS}}
        except (KeyError, ValueError) as exc:
            raise ProducerError(f"bad banner: {exc}") from exc

    def _region(self, node: Node, page: str | None = None) -> dict:
        if node.role not in ("region", "navigation") or node.name is None:
            raise ProducerError(f"unmapped node: {node.role} {node.name!r}")
        title = node.name
        if title.startswith(catalog.CHANGED_PREFIX):
            base = catalog.REGION_BY_TITLE.get(title[len(catalog.CHANGED_PREFIX):])
            role = f"changed:{base.role}" if base else None
        else:
            base = catalog.REGION_BY_TITLE.get(title)
            role = base.role if base else None
        if base is None:
            raise ProducerError(f"region not in catalog: {title!r}")
        records, controls = [], []
        pending_label: str | None = None
        for child in self._flatten(node.children):
            if child.role == "heading" and child.name == title:
                continue
            if child.role == "separator" and base.role == "actions":
                continue  # visual divider between action groups
            if child.role == "table" and base.record_type:
                records = self._table(child, base)
            elif child.role == "text" and base.role == "actions":
                pending_label = _text_of(child)
            elif child.role in ("button", "link", "textbox", "combobox"):
                if child.role == "textbox":
                    if pending_label != child.name:
                        raise ProducerError(f"textbox {child.name!r} without its visible label")
                    pending_label = None
                controls.append(self._control(node, child))
            else:
                raise ProducerError(f"unmapped node in region {title!r}: {child.role} {child.name!r}")
        if pending_label is not None:
            raise ProducerError(f"unmapped text in region {title!r}: {pending_label!r}")
        if base.role == "navigation":
            self._check_navigation_state(node, controls, page)
        return {"role": role, "records": records, "controls": controls}

    def _check_navigation_state(self, nav: Node, controls: list[dict], page: str | None) -> None:
        """Exactly one link is aria-current="page", and it is the page in the banner."""
        scope = self.page.get_by_role("navigation", name=nav.name, exact=True)
        current = [c["name"] for c in controls
                   if scope.get_by_role("link", name=c["name"], exact=True).get_attribute("aria-current") == "page"]
        if page is None or current != [catalog.PAGE_TITLE[page]]:
            raise ProducerError(f"navigation state {current} does not match the page {page!r}")

    @staticmethod
    def _flatten(children: list[Node]) -> list[Node]:
        """Lists, list items and a link's /url property are structure only (not visible content)."""
        out = []
        for c in children:
            if c.role in ("list", "listitem") and c.value is None:
                out.extend(BrowserSurface._flatten(c.children))
            elif c.role == "link":
                if any(g.role != "/url" for g in c.children):
                    raise ProducerError(f"unexpected content inside link {c.name!r}")
                out.append(Node("link", c.name, c.attrs, None))
            else:
                out.append(c)
        return out

    def _table(self, table: Node, base: catalog.Region) -> list[dict]:
        rows = [r for g in table.children if g.role == "rowgroup" for r in g.children]
        if any(g.role != "rowgroup" for g in table.children) or any(r.role != "row" for r in rows):
            raise ProducerError(f"unexpected table structure in {base.title!r}")
        expected = ["Record", "Rev"] + [c.header for c in base.columns]
        header = [r for r in rows if r.children and all(c.role == "columnheader" for c in r.children)]
        if len(header) != 1 or [normalize_text(c.name or "") for c in header[0].children] != expected:
            raise ProducerError(f"column headers of {base.title!r} do not match the catalog")
        body = [row for row in rows if row is not header[0]]
        if len(body) == 1 and [_text_of(c) for c in body[0].children] == [catalog.EMPTY_MESSAGE]:
            return []  # the empty-table message (adapted from eagleview-rpa's table-message rows)
        records = []
        for row in body:
            cells = [_text_of(c) for c in row.children if c.role == "cell"]
            if len(cells) != len(expected) or len(cells) != len(row.children):
                raise ProducerError(f"row shape mismatch in {base.title!r}")
            try:
                fields = {col.field: parse_cell(col, text) for col, text in zip(base.columns, cells[2:])}
                version = int(cells[1])
            except (ParseError, ValueError) as exc:
                raise ProducerError(f"{base.title}: {exc}") from exc
            records.append({"id": cells[0], "type": base.record_type, "version": version, "fields": fields})
        return records

    def _control(self, region: Node, c: Node) -> dict:
        scope = self.page.get_by_role(region.role, name=region.name, exact=True)
        loc = scope.get_by_role(c.role, name=c.name, exact=True)
        if loc.count() != 1:
            raise ProducerError(f"control {c.role} {c.name!r} is not unique in {region.name!r}")
        cid = loc.get_attribute("data-agent-id")
        if not cid:
            raise ProducerError(f"control {c.name!r} has no data-agent-id")
        out = {"id": cid, "role": c.role, "name": normalize_text(c.name or ""), "enabled": "disabled" not in c.attrs}
        if c.role == "textbox":
            out["value"] = c.value or ""
        return out

