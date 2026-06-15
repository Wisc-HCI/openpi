#!/usr/bin/env python3
"""Build and optionally serve a lightweight static preview of this docs site.

The source of truth is still the Jekyll/Just-the-Docs markdown tree. This script
exists so the docs can be previewed on machines without Ruby development headers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_site"


@dataclass(frozen=True)
class Page:
    source: Path
    title: str
    nav_order: int
    permalink: str
    parent: str | None
    has_children: bool
    body: str


def parse_front_matter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text

    _, front, body = text.split("---", 2)
    data: dict[str, str] = {}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data, body.lstrip()


def load_pages() -> list[Page]:
    pages: list[Page] = []
    for path in sorted(ROOT.rglob("*.md")):
        if path.name == "README.md" or "_site" in path.parts or "vendor" in path.parts:
            continue
        front, body = parse_front_matter(path)
        if front.get("layout") != "default":
            continue
        title = front.get("title", path.stem.replace("-", " ").title())
        permalink = front.get("permalink") or "/" + path.with_suffix("").relative_to(ROOT).as_posix() + "/"
        pages.append(
            Page(
                source=path,
                title=title,
                nav_order=int(front.get("nav_order", "999")),
                permalink=permalink,
                parent=front.get("parent") or None,
                has_children=front.get("has_children", "false").lower() == "true",
                body=body,
            )
        )
    return sorted(pages, key=lambda page: (page.parent or "", page.nav_order, page.title))


def inline_markdown(text: str) -> str:
    placeholders: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        placeholders.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash_code, html.escape(text))
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_replace, text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def link_replace(match: re.Match[str]) -> str:
    label = match.group(1)
    href = match.group(2).replace("{{ site.baseurl }}", "")
    return f'<a href="{html.escape(href)}">{label}</a>'


def render_markdown(markdown: str) -> str:
    lines = markdown.splitlines()
    html_lines: list[str] = []
    index = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            html_lines.append("<p>" + inline_markdown(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped.strip("`").strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            class_name = f' class="language-{html.escape(language)}"' if language else ""
            html_lines.append(f"<pre><code{class_name}>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            content = stripped[level:].strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", content.lower()).strip("-")
            html_lines.append(f'<h{level} id="{anchor}">{inline_markdown(content)}</h{level}>')
            index += 1
            continue

        if is_table_start(lines, index):
            flush_paragraph()
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            html_lines.append(render_table(table_lines))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items: list[str] = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[index].strip()))
                index += 1
            html_lines.append("<ol>" + "".join(f"<li>{inline_markdown(item)}</li>" for item in items) + "</ol>")
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:])
                index += 1
            html_lines.append("<ul>" + "".join(f"<li>{inline_markdown(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return "\n".join(html_lines)


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return lines[index].strip().startswith("|") and bool(re.match(r"^\|?[\s:-]+\|", lines[index + 1].strip()))


def render_table(lines: list[str]) -> str:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    header = rows[0]
    body = rows[2:]
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{inline_markdown(cell)}</th>" for cell in header)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{inline_markdown(cell)}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_nav(pages: list[Page], current: Page) -> str:
    by_parent: dict[str | None, list[Page]] = {}
    for page in pages:
        by_parent.setdefault(page.parent, []).append(page)
    for siblings in by_parent.values():
        siblings.sort(key=lambda page: (page.nav_order, page.title))

    def render_items(parent: str | None) -> str:
        items = []
        for page in by_parent.get(parent, []):
            active = " active" if page.permalink == current.permalink else ""
            child_html = render_items(page.title)
            items.append(
                f'<li><a class="{active.strip()}" href="{page.permalink}">{html.escape(page.title)}</a>{child_html}</li>'
            )
        return "<ul>" + "".join(items) + "</ul>" if items else ""

    return render_items(None)


def layout(page: Page, pages: list[Page]) -> str:
    nav = build_nav(pages, page)
    content = render_markdown(page.body)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page.title)} - DROID/OpenPI Lab Pipeline</title>
  <link rel="stylesheet" href="/assets/site.css">
</head>
<body>
  <aside>
    <a class="brand" href="/">DROID/OpenPI Lab Pipeline</a>
    <nav>{nav}</nav>
  </aside>
  <main>
    {content}
  </main>
</body>
</html>
"""


CSS = """
:root { --text:#172033; --muted:#647083; --line:#d9dee8; --side:#f7f8fa; --link:#2563eb; --code:#eef2f6; }
* { box-sizing: border-box; }
body { margin:0; color:var(--text); font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; display:grid; grid-template-columns:300px minmax(0,1fr); min-height:100vh; }
aside { background:var(--side); border-right:1px solid var(--line); padding:24px 18px; position:sticky; top:0; height:100vh; overflow:auto; }
main { max-width:900px; padding:44px 56px 80px; }
.brand { display:block; color:var(--text); font-weight:800; text-decoration:none; margin:0 0 22px; }
nav ul { list-style:none; padding-left:0; margin:0; }
nav ul ul { padding-left:14px; margin:2px 0 8px; }
nav a { display:block; padding:6px 9px; border-radius:6px; color:#334155; text-decoration:none; font-size:14px; }
nav a:hover, nav a.active { background:#eaf1ff; color:var(--link); }
h1 { font-size:42px; line-height:1.08; margin:0 0 20px; }
h2 { border-top:1px solid var(--line); padding-top:28px; margin-top:34px; }
h3 { margin-top:26px; }
a { color:var(--link); }
code { background:var(--code); border-radius:5px; padding:0.1em 0.35em; }
pre { background:#111827; color:#eef2ff; padding:16px; border-radius:8px; overflow:auto; }
pre code { background:transparent; padding:0; }
table { border-collapse:collapse; width:100%; margin:16px 0 24px; }
th, td { border:1px solid var(--line); padding:10px 12px; vertical-align:top; }
th { background:var(--side); text-align:left; font-size:13px; }
@media (max-width: 860px) { body { display:block; } aside { position:static; height:auto; } main { padding:28px 22px 56px; } h1 { font-size:34px; } }
"""


def write_page(page: Page, pages: list[Page]) -> None:
    target_dir = OUT / page.permalink.strip("/")
    if page.permalink == "/":
        target_dir = OUT
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "index.html").write_text(layout(page, pages), encoding="utf-8")


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    (OUT / "assets" / "site.css").write_text(CSS, encoding="utf-8")
    pages = load_pages()
    for page in pages:
        write_page(page, pages)
    print(f"Built {len(pages)} pages into {OUT}")


def serve(port: int) -> None:
    build()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(OUT), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving http://127.0.0.1:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Build and serve the static preview.")
    parser.add_argument("--port", type=int, default=4000)
    args = parser.parse_args()
    if args.serve:
        serve(args.port)
    else:
        build()


if __name__ == "__main__":
    main()
