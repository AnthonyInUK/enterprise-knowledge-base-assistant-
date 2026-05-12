from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "sources" / "energy_sources.json"
OUTPUT_DIR = ROOT / "data" / "raw" / "energy"


@dataclass
class CollectedArtifact:
    source_id: str
    title: str
    source_url: str
    artifact_type: str
    local_path: str
    content_type: str
    sha256: str
    discovered_from: str | None
    downloaded_at: str
    company: str | None = None
    organization: str | None = None
    region: str | None = None
    category: str | None = None
    tags: list[str] | None = None


class PdfLinkParser(HTMLParser):
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
        if ".pdf" in absolute.lower():
            self.links.append(absolute)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fetch_url(url: str) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Codex RAG data collector)",
            "Accept": "*/*",
        },
    )
    with urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get_content_type()
        return content, content_type


def write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def collect_pdf_links(html: bytes, base_url: str) -> list[str]:
    parser = PdfLinkParser(base_url)
    parser.feed(html.decode("utf-8", errors="ignore"))
    return list(dict.fromkeys(parser.links))


def infer_artifact_name(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        return fallback
    return name


def collect_source(source: dict[str, object]) -> list[CollectedArtifact]:
    source_id = str(source["id"])
    title = str(source["title"])
    source_url = str(source["source_url"])
    source_kind = str(source.get("source_kind", "page"))
    company = str(source.get("company") or source.get("organization") or "unknown")
    organization = source.get("organization")
    region = source.get("region")
    category = source.get("category")
    tags = source.get("tags")

    source_dir = OUTPUT_DIR / slugify(company) / slugify(source_id)
    source_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[CollectedArtifact] = []
    downloaded_at = datetime.now(timezone.utc).isoformat()

    if source_kind == "pdf":
        content, content_type = fetch_url(source_url)
        filename = infer_artifact_name(source_url, f"{source_id}.pdf")
        local_path = source_dir / filename
        write_file(local_path, content)
        artifacts.append(
            CollectedArtifact(
                source_id=source_id,
                title=title,
                source_url=source_url,
                artifact_type="pdf",
                local_path=str(local_path.relative_to(ROOT)),
                content_type=content_type,
                sha256=sha256_bytes(content),
                discovered_from=None,
                downloaded_at=downloaded_at,
                company=str(source.get("company")) if source.get("company") else None,
                organization=str(organization) if organization else None,
                region=str(region) if region else None,
                category=str(category) if category else None,
                tags=list(tags) if isinstance(tags, list) else None,
            )
        )
        return artifacts

    html, content_type = fetch_url(source_url)
    page_path = source_dir / "index.html"
    write_file(page_path, html)
    artifacts.append(
        CollectedArtifact(
            source_id=source_id,
            title=title,
            source_url=source_url,
            artifact_type="html",
            local_path=str(page_path.relative_to(ROOT)),
            content_type=content_type,
            sha256=sha256_bytes(html),
            discovered_from=None,
            downloaded_at=downloaded_at,
            company=str(source.get("company")) if source.get("company") else None,
            organization=str(organization) if organization else None,
            region=str(region) if region else None,
            category=str(category) if category else None,
            tags=list(tags) if isinstance(tags, list) else None,
        )
    )

    pdf_links = collect_pdf_links(html, source_url)
    for index, pdf_url in enumerate(pdf_links, start=1):
        pdf_content, pdf_content_type = fetch_url(pdf_url)
        filename = infer_artifact_name(pdf_url, f"{source_id}-{index}.pdf")
        pdf_path = source_dir / "pdfs" / filename
        write_file(pdf_path, pdf_content)
        artifacts.append(
            CollectedArtifact(
                source_id=source_id,
                title=title,
                source_url=pdf_url,
                artifact_type="pdf",
                local_path=str(pdf_path.relative_to(ROOT)),
                content_type=pdf_content_type,
                sha256=sha256_bytes(pdf_content),
                discovered_from=source_url,
                downloaded_at=downloaded_at,
                company=str(source.get("company")) if source.get("company") else None,
                organization=str(organization) if organization else None,
                region=str(region) if region else None,
                category=str(category) if category else None,
                tags=list(tags) if isinstance(tags, list) else None,
            )
        )

    return artifacts


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest not found: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    collected: list[CollectedArtifact] = []

    for source in manifest:
        try:
            artifacts = collect_source(source)
            collected.extend(artifacts)
            print(f"collected {len(artifacts)} artifacts from {source['id']}")
        except Exception as exc:  # noqa: BLE001
            print(f"failed to collect {source['id']}: {exc}")

    summary_path = OUTPUT_DIR / "_collection_manifest.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        for artifact in collected:
            fh.write(json.dumps(asdict(artifact), ensure_ascii=False) + "\n")

    print(f"wrote {len(collected)} artifacts to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
