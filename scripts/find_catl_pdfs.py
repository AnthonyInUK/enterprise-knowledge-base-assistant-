from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_SEEDS = [
    "https://www.catl.com/en/",
    "https://www.catl.com/en/news/6773.html",
    "https://www.catl.com/en/news/6392.html",
    "https://www.catl.com/en/search?keyword=annual%20report",
    "https://www.catl.com/en/investor",
    "https://www.catl.com/en/investors",
    "https://www.catl.com/en/ir",
    "https://www.catl.com/en/sitemap.xml",
    "https://www.catl.com/sitemap.xml",
]

PDF_RE = re.compile(r"\.pdf(?:\?.*)?$", re.IGNORECASE)
SITEMAP_LOC_RE = re.compile(r"<loc>(.*?)</loc>", re.IGNORECASE)


@dataclass
class DiscoveredPdf:
    pdf_url: str
    discovered_from: str


class LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        absolute = urljoin(self.base_url, href)
        self.links.append(absolute)


def fetch_url(url: str) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Codex CATL PDF Finder)",
            "Accept": "text/html,application/pdf,*/*",
        },
    )
    with urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get_content_type()
        return content, content_type


def is_same_site(url: str, seed_host: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == seed_host


def extract_links(base_url: str, html: bytes) -> list[str]:
    parser = LinkParser(base_url)
    parser.feed(html.decode("utf-8", errors="ignore"))
    return list(dict.fromkeys(parser.links))


def extract_sitemap_links(xml_bytes: bytes) -> list[str]:
    text = xml_bytes.decode("utf-8", errors="ignore")
    links = [match.strip()
             for match in SITEMAP_LOC_RE.findall(text) if match.strip()]
    return list(dict.fromkeys(links))


def iter_pdfs(
    seeds: Iterable[str],
    max_pages: int,
    max_depth: int,
    sleep_s: float,
) -> list[DiscoveredPdf]:
    queue: list[tuple[str, int, str]] = []
    visited: set[str] = set()
    discovered: dict[str, DiscoveredPdf] = {}

    for seed in seeds:
        queue.append((seed, 0, seed))

    while queue and len(visited) < max_pages:
        url, depth, parent = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            content, content_type = fetch_url(url)
        except Exception:
            continue

        if PDF_RE.search(url) or content_type == "application/pdf":
            if url not in discovered:
                discovered[url] = DiscoveredPdf(
                    pdf_url=url, discovered_from=parent)
            continue

        if content_type in {"application/xml", "text/xml"} or url.lower().endswith(".xml"):
            if depth < max_depth:
                for link in extract_sitemap_links(content):
                    if link not in visited:
                        queue.append((link, depth + 1, url))
            continue

        if depth >= max_depth:
            continue

        seed_host = urlparse(url).netloc
        links = extract_links(url, content)
        for link in links:
            if link in visited:
                continue
            if PDF_RE.search(link):
                if link not in discovered:
                    discovered[link] = DiscoveredPdf(
                        pdf_url=link, discovered_from=url)
                continue
            if is_same_site(link, seed_host):
                queue.append((link, depth + 1, url))

        if sleep_s > 0:
            time.sleep(sleep_s)

    return list(discovered.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find CATL annual report PDFs on catl.com.")
    parser.add_argument("--seed", action="append", default=DEFAULT_SEEDS)
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument(
        "--output", default="data/raw/energy/catl/catl_pdf_links.jsonl")
    args = parser.parse_args()

    results = iter_pdfs(args.seed, args.max_pages, args.max_depth, args.sleep)
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as fh:
        for item in results:
            fh.write(
                json.dumps(
                    {
                        "pdf_url": item.pdf_url,
                        "discovered_from": item.discovered_from,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"found {len(results)} pdf links")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
