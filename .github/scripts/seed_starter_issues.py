"""Preview or publish curated issues. Idempotent across open AND closed issues.

Preview: python3 .github/scripts/seed_starter_issues.py
Publish: python3 .github/scripts/seed_starter_issues.py --apply
Uses GH_TOKEN/GITHUB_TOKEN or the already authenticated GitHub CLI, never prints it.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from contributors import DEFAULT_MAINTAINERS, GitHub, maintainers

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "good first issue": ("7057ff", "A scoped task suitable for a first contribution"),
    "help wanted": ("008672", "Contributions welcome"),
    "tests": ("1d76db", "Test coverage and regression protection"),
    "documentation": ("0075ca", "Documentation improvements"),
    "bug": ("d73a4a", "Something is not working as expected"),
}


def load_issues():
    issues = json.loads((ROOT / "starter-issues.json").read_text())
    ids = [item["id"] for item in issues]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate starter issue IDs")
    return issues


def issue_body(issue, owner):
    return (f"<!-- cyberwave-starter:{issue['id']} -->\n"
            f"**Mentor:** @{owner}\n\n{issue['body']}")


def seed(github, issues, owner):
    # Validate before creating anything. Mentors are mentioned, not assigned:
    # the eventual contributor should be the issue assignee.
    permission = github.api(f"collaborators/{owner}/permission")
    if permission.get("permission") not in ("admin", "write", "maintain"):
        raise ValueError("The starter-issue mentor must have write access to this repository.")
    existing = list(github.pages("issues?state=all"))
    labels = {item["name"] for item in github.pages("labels")}
    for label in sorted({label for issue in issues for label in issue["labels"]}):
        if label not in labels:
            color, description = LABELS[label]
            github.api("labels", method="POST", data={"name": label, "color": color, "description": description})
    urls = []
    for issue in issues:
        marker = f"<!-- cyberwave-starter:{issue['id']} -->"
        match = next((item for item in existing if "pull_request" not in item and
                      (marker in (item.get("body") or "") or item["title"] == issue["title"])), None)
        if match:
            urls.append(match["html_url"])
            continue
        created = github.api("issues", method="POST", data={
            "title": issue["title"], "body": issue_body(issue, owner), "labels": issue["labels"],
        })
        urls.append(created["html_url"])
    return urls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create labels and issues on GitHub")
    args = parser.parse_args()
    issues = load_issues()
    owner = maintainers(os.getenv("SDK_MAINTAINERS") or DEFAULT_MAINTAINERS)[0]
    if not args.apply:
        for issue in issues:
            print(f"# {issue['title']}\n\n{issue_body(issue, owner)}\n")
        return
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        token = subprocess.run(["gh", "auth", "token"], check=True, capture_output=True, text=True).stdout.strip()
    github = GitHub("cyberwave-os/cyberwave-python", token)
    for url in seed(github, issues, owner):
        print(url)


if __name__ == "__main__":
    main()
