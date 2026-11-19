"""Local HTTP server for the browser surface and for manual play (stdlib only).

GET  /view          render the current page (no side effect; safe to reload)
GET  /go/<page>     navigate (charges one action) then 303 -> /view
GET  /ui/observe    re-render the current page (charges one action) then 303 -> /view
POST /ui/draft      store the typed quantity (charges one action) then 303 -> /view
POST /op/<key>      submit a case action to the engine (charges one action) then 303 -> /view
POST /op/posting    a posting button with the quantity box's value (fields op, quantity): one action
POST /manual/advance?t=<token>  manual play only: let business time pass (agents use the advance_time
                    tool). The token is issued with each rendered page and is single-use, so a
                    refresh, a back-button resubmission or an older tab cannot repeat an advance.

With `limits` set (manual play), every charging request passes the same horizon and action-limit rules
as the agent's tool layer; a refused request returns 409 and changes no record. Agent surfaces leave
`limits` unset because the tool layer has already applied those rules.
"""
from __future__ import annotations

import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ap_sim.app import catalog
from ap_sim.app.controls import control_ids
from ap_sim.app.limits import EpisodeLimits
from ap_sim.app.render import render_page
from ap_sim.app.session import AppSession
from ap_sim.app.viewmodel import build_page
from ap_sim.events.clock import ACTION_COST, format_time


class AppServer:
    def __init__(self, session: AppSession):
        self.session = session
        self.lock = threading.Lock()
        self.on_render = None  # optional callback(session) after each /view render (manual-play log)
        self.limits: EpisodeLimits | None = None  # manual play only
        self.refusals: list[dict] = []
        self.advance_token: str | None = None  # manual play: the single-use token of the latest page
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _redirect(self):
                self.send_response(303)
                self.send_header("Location", "/view")
                self.end_headers()

            def _not_found(self):
                self.send_response(404)
                self.end_headers()

            def _refuse(self, s, reason, request):
                """Manual play: a request the limits refuse costs one action (capped at the horizon), changes
                no record, and is logged."""
                if reason == "horizon":  # like the tool layer: a horizon refusal costs time; a call past the limit does not
                    s.charge(app.limits.refusal_cost(s.engine.clock))
                app.refusals.append({"request": request, "reason": reason, "at": s.engine.clock})
                text = {"horizon": f"Refused: the business horizon ({format_time(app.limits.horizon)}) is reached "
                                   "or this action cannot complete before it.",
                        "action_limit": f"Refused: the action limit ({app.limits.action_limit}) is reached."}[reason]
                body = (text + " No business action was taken. Press Enter in the play terminal to finish.\n").encode()
                self.send_response(409)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _refuse_without_charge(self, reason):
                """advance_time at the horizon: refused like the tool layer's, with no time to charge."""
                body = (f"Refused: {reason}. No business action was taken. Press Enter in the play terminal "
                        "to finish.\n").encode()
                self.send_response(409)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _manual_advance(self, s, url, form):
                if app.limits is None:
                    return self._not_found()
                token = parse_qs(url.query).get("t", [""])[0]
                if not app.advance_token or token != app.advance_token:  # stale page: never advance twice
                    app.refusals.append({"request": url.path, "reason": "stale", "at": s.engine.clock})
                    return self._redirect()
                app.advance_token = None  # single use
                try:
                    minutes = int(form.get("minutes", ["0"])[0])
                except ValueError:
                    minutes = 0
                if not 1 <= minutes <= 480:
                    return self._not_found()
                reason, minutes = app.limits.admit_advance(s.engine.clock, minutes)
                if reason is not None:
                    app.refusals.append({"request": url.path, "reason": reason, "at": s.engine.clock})
                    return self._refuse_without_charge(reason)
                s.advance(minutes)
                return self._redirect()

            def _admit(self, s, request, cost=ACTION_COST):
                if app.limits is None:
                    return True
                reason = app.limits.admit(s.engine.clock, cost)
                if reason is not None:
                    self._refuse(s, reason, request)
                    return False
                return True

            def do_GET(self):
                url = urlparse(self.path)
                with app.lock:
                    s = app.session
                    if url.path == "/view":
                        action = None
                        if app.limits is not None:  # manual play: issue a fresh single-use advance token
                            app.advance_token = secrets.token_hex(8)
                            action = f"/manual/advance?t={app.advance_token}"
                        body = render_page(build_page(s), control_ids(s.seed), action).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        if app.on_render is not None:
                            app.on_render(s)
                        return
                    if url.path.startswith("/go/") and url.path[4:] in catalog.pages_for(s.schema):   # only this session's pages
                        if not self._admit(s, url.path):
                            return
                        s.navigate(url.path[4:])
                    elif url.path == "/ui/observe":
                        if not self._admit(s, url.path):
                            return
                        s.observe()
                    else:
                        return self._not_found()
                self._redirect()

            def do_POST(self):
                url = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                with app.lock:
                    s = app.session
                    if url.path == "/manual/advance":
                        return self._manual_advance(s, url, form)
                    if not self._admit(s, url.path):
                        return
                    if url.path == "/ui/draft":
                        s.fill(form.get("value", [""])[0])
                    elif url.path == "/op/posting" and form.get("op", [""])[0] in catalog.QUANTITY_ACTIONS:
                        s.submit_with_quantity(form["op"][0], form.get("quantity", [""])[0])
                    elif url.path.startswith("/op/") and url.path[4:] in catalog.ACTION \
                            and catalog.ACTION[url.path[4:]][1] == "button":
                        s.operate(url.path[4:])
                    else:
                        return self._not_found()
                self._redirect()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "AppServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
