"""Check built documentation links and assets using only the standard library."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys


class Links(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.links = []
        self.ids = set()
        self.canonical = ""
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href", "")
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve()
    pages = {path: Links(path.read_text()) for path in root.rglob("*.html")}
    if not pages:
        raise SystemExit(f"No HTML pages in {root}; run mkdocs build first")
    base_path = urlsplit(pages[root / "index.html"].canonical).path
    errors = []
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            local_path = unquote(url.path)
            if local_path.startswith("/") and local_path.startswith(base_path):
                target = (root / local_path[len(base_path) :]).resolve()
            else:
                target = (path.parent / local_path).resolve() if local_path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f"{path.relative_to(root)}: missing {link}")
            elif (
                url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(root)}: missing anchor {link}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated local links and assets across {len(pages)} HTML pages.")


if __name__ == "__main__":
    main()
