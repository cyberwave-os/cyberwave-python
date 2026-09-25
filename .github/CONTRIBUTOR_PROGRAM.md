# Cyberwave SDK Contributors: maintainer runbook

The MVP welcomes human PR authors once, requests a named maintainer review,
posts an advisory AI review, and celebrates merged contributions in Discord.
It also ships four curated, hardware-free starter issues ready to publish.

Primary maintainer: **@khushisharma22**. No backup is configured yet. PRs authored
by this maintainer need another human reviewer selected manually; the automation
never requests a review from the PR's author.

## Activate

1. Review and merge these files to the default branch (`main`). The privileged
   workflow always checks out scripts from the default branch, including PRs
   targeting `dev`. It does not run the version in the incoming PR.
2. In **Settings → Secrets and variables → Actions**, add these repository secrets:
   - `SDK_ANTHROPIC_API_KEY`: an Anthropic API key with access to the configured
     Claude model. Use a dedicated workspace with appropriate usage limits.
     Never put the key in a PR, issue, chat message, or committed file.
   - `SDK_DISCORD_WEBHOOK_URL`: an incoming webhook for the intended Discord
     contribution channel, in the form `https://discord.com/api/webhooks/ID/TOKEN`.
3. Optional repository variables:
   - `SDK_MAINTAINERS`: comma-separated GitHub usernames in priority order;
     defaults to `khushisharma22`. Add a backup here and in `CODEOWNERS` when ready.
   - `SDK_REVIEW_MODEL`: defaults to `claude-sonnet-4-6`. Overrides must support
     the Anthropic Messages API and `output_config.format` JSON Schema output.
4. Confirm `khushisharma22` has write access and is eligible for review requests.
   The workflow grants only `contents: read` and `pull-requests: write` to its
   jobs. It has no permission to push commits or publish releases.
5. Check the repository/organization's **Actions event policies**. This workflow
   requires `pull_request_target` for fork-compatible comments and secret access.
   GitHub's default policy for public repositories is scheduled for enforcement
   on November 2, 2026. If necessary, allow this specific trusted workflow under
   your organization's policy rather than enabling arbitrary privileged PR code.
   See [GitHub's current guidance](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target).
6. Publish the curated issues after reviewing their preview:

   ```bash
   python3 .github/scripts/seed_starter_issues.py
   python3 .github/scripts/seed_starter_issues.py --apply
   ```

   Publishing uses `GH_TOKEN`, `GITHUB_TOKEN`, or the authenticated GitHub CLI.
   It creates missing labels and skips matching issues, including closed ones.
   It mentions the mentor, leaving assignment available for the contributor.
   The preview performs no network calls. Revisit the backlog before republishing
   from an old checkout; tasks may already have been solved.

Missing credentials do not produce fake success: an unavailable review is stated
in the review comment, and a missing Discord webhook is recorded in the Actions
summary. Add the missing secret and manually dispatch **SDK contributors** from
`main` with the PR number to recover. No historical PRs are processed automatically.

## Contributor experience

- Opening a PR triggers a single welcome comment with help links and maintainer
  contacts. Reopening or pushing does not add another welcome.
- Drafts receive a welcome. Review requests and AI review wait until the PR is
  marked ready. Existing requested reviewers/teams are respected.
- The first configured maintainer who is not the author is requested. If the
  request fails, configured backups are tried. Completed maintainer reviews
  are not repeatedly requested on every push.
- Ready PRs receive one updatable **Automated advisory review** comment. The
  model cannot approve, request changes, execute tools, or merge a PR.
- New commits cause another advisory review, updating that comment. Results
  for a stale commit, closed PR, or newly drafted PR are discarded.
- A merged human PR receives a GitHub thank-you and a Discord message containing
  the PR title, URL, and GitHub username. Bots are excluded.
- Contributors may check the opt-out box in the PR template, add
  `<!-- cyberwave:no-celebration -->` to their PR body, or ask a maintainer to add
  the `no-celebration` label before merge. Discord identity linking is not needed.

## Review scope, cost, and data

Eligible patches are sent to the Anthropic Messages API for a Claude review.
No repository credentials, workflow environment, PR body, or external tool access
are provided to the model. Anthropic's applicable API data controls govern
processing; the integration does not claim zero data retention.

### Customize the review

Edit [review-instructions.md](review-instructions.md) to change Cyberwave's review
priorities, conventions, or evidence requirements. The supplied policy covers
public API compatibility, optional dependencies, connection cleanup, simulation
versus live robotics behavior, and useful regression tests.

Only the policy on the **default branch** is used. A PR modifying the policy is
reviewed under the existing policy; its changes apply after merge. The fixed
response schema, three-finding cap, and no-tool execution boundary stay in the
script. To re-review an already reviewed commit after changing the policy or
model, delete only the bot's advisory review comment and dispatch that PR again.
Otherwise the successful head-commit deduplication intentionally skips it.

Each review sends at most **40 files / 60,000 patch characters** and requests at
most **1,800 output tokens**, with at most three actionable findings. Generated
`cyberwave/rest/` files, lockfiles, notebooks, SVGs, and unavailable patches are
excluded. Other files exceeding the budget are skipped rather than silently
cut mid-patch. GitHub itself may truncate individual patches; the comment states
the coverage and limitations. No full-repository correctness guarantee is made.

Review calls are deduplicated by head commit after a successful response. Failed
or missing-key reviews can be retried with **Run workflow**. There is no global
monthly spend cap in the script: configure the Anthropic workspace's spend controls
before launch. Rapid pushes are coalesced by GitHub's per-PR concurrency queue;
an active run is not canceled during comment delivery.

Provider errors leave an advisory unavailable comment. Do not make **SDK
contributors / advisory-review** a required branch-protection check. Keep the
existing SDK CI checks and maintainer approval requirements in force.

References: [Claude structured output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
[Messages API](https://platform.claude.com/docs/en/api/messages/create),
[model reference](https://platform.claude.com/docs/en/models/sonnet-4-6/overview).

## Trust boundaries

- Privileged jobs check out only the trusted default branch, with credential
  persistence disabled and no SDK dependency installation or cache restore.
- Changed PR files are fetched through GitHub's API as JSON data; they are never
  imported, checked out, built, or executed in privileged jobs.
- The independent `Contributor automation tests` workflow runs on `pull_request`
  without secrets and with a read-only token.
- PR numbers enter through environment variables and are parsed as positive
  integers. PR titles, patches, and usernames never enter shell commands.
- Only the Actions bot's comments with the exact marker prefix count as state.
  A contributor cannot bypass the automation by posting a matching marker.
- HTTP redirects are refused. Errors omit response bodies and credential-bearing
  URLs. Discord messages disable mentions and escape untrusted formatting.
- Workflows run only in `cyberwave-os/cyberwave-python`, targeting `main` or `dev`.
  A manual dispatch must use the default branch.

## Discord delivery and recovery

GitHub and Discord do not share a transaction. This MVP prioritizes avoiding
duplicate celebrations over blindly retrying uncertain sends. It is not a claim
of exactly-once delivery under every network failure.

Before sending, the workflow creates one merge thank-you comment containing
`<!-- discord:pending -->`. It sends with `wait=true` and records the returned
Discord message ID in that same comment as `<!-- discord:sent:ID -->`.
Per-PR concurrency serializes community runs. Repeated events or manual reruns
do not send again if either marker exists.

If the process crashes, Discord times out, or recording the ID fails:

1. Inspect the contribution channel for the PR URL and the Actions run summary.
2. If the message exists, edit the merge thank-you comment's source and replace
   the pending marker with `<!-- discord:sent:MESSAGE_ID -->` using its real ID.
3. If absence is confirmed, delete **only the bot's merge thank-you comment**,
   then dispatch **SDK contributors** from `main` with that PR number. Never
   clear the marker while the original run is still active.

This manual reconciliation also applies to Discord rate limiting. Reads retry
transient failures; writes are deliberately not automatically repeated.
See [Discord webhook semantics](https://docs.discord.com/developers/resources/webhook).

## Validation before launch

Local tests need Python 3.10+ and no credentials:

```bash
python3 -m unittest discover -s .github/scripts/tests -v
git diff --check
```

After merging and configuring secrets, use a disposable PR from a fork:

1. Open as draft: one welcome, no AI review/request yet.
2. Mark ready: request the named maintainer and publish one advisory review.
3. Push an update and rerun: one welcome and one review comment remain.
4. Close without merging: no Discord celebration.
5. Merge an eligible PR: one Discord message and a recorded message ID.
6. Rerun that merged PR: no duplicate. Verify opt-out using a separate PR.

The local tests mock GitHub, Claude, and Discord. Live delivery and model quality
must still be verified after activation. No live credentials are needed to review
or test this implementation locally.
