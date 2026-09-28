"""Behavioral tests for public-PR automation; all external calls are faked."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import contributors as app
import seed_starter_issues as starter


def pull(**overrides):
    pr = {
        "number": 42, "state": "open", "draft": False, "merged": False,
        "user": {"login": "new-contributor", "type": "User"},
        "head": {"sha": "a" * 40}, "title": "Improve connection cleanup",
        "base": {"ref": "main", "repo": {"full_name": "cyberwave-os/cyberwave-python"}},
        "body": "", "labels": [], "requested_reviewers": [], "requested_teams": [],
    }
    pr.update(overrides)
    return pr


def file_change(path="cyberwave/client.py", patch_text="@@ -1 +1 @@\n-old\n+new"):
    return {"filename": path, "patch": patch_text, "status": "modified"}


class FakeGitHub(app.GitHub):
    def __init__(self, pr=None):
        super().__init__("cyberwave-os/cyberwave-python", "fake-token")
        self.pr = pr or pull()
        self.records = []
        self.writes = []
        self.files = [file_change()]
        self.reviews = []
        self.issues = []
        self.labels = []
        self.fail_reviewers = set()

    def pages(self, path):
        if path.endswith("/comments"):
            return copy.deepcopy(self.records)
        if path.endswith("/files"):
            return copy.deepcopy(self.files)
        if path.endswith("/reviews"):
            return copy.deepcopy(self.reviews)
        if path == "issues?state=all":
            return copy.deepcopy(self.issues)
        if path == "labels":
            return copy.deepcopy(self.labels)
        raise AssertionError(f"Unexpected paginated read: {path}")

    def api(self, path, *, method="GET", data=None):
        if method == "GET":
            if path == "pulls/42":
                return copy.deepcopy(self.pr)
            if path.startswith("collaborators/"):
                return {"permission": "admin"}
            raise AssertionError(f"Unexpected read: {path}")
        self.writes.append((method, path, copy.deepcopy(data)))
        if path == "pulls/42/requested_reviewers":
            if data["reviewers"][0] in self.fail_reviewers:
                raise app.ServiceError("Reviewer unavailable")
            self.pr["requested_reviewers"] = [{"login": name} for name in data["reviewers"]]
            return self.pr
        if path == "issues/42/comments":
            comment = {"id": len(self.records) + 1, "body": data["body"],
                       "user": {"login": app.BOT, "type": "Bot"}}
            self.records.append(comment)
            return copy.deepcopy(comment)
        if path.startswith("issues/comments/"):
            comment = next(item for item in self.records if item["id"] == int(path.split("/")[-1]))
            comment["body"] = data["body"]
            return copy.deepcopy(comment)
        if path == "labels":
            self.labels.append(data)
            return data
        if path == "issues":
            issue = dict(data, html_url=f"https://github.com/example/issues/{len(self.issues) + 1}")
            self.issues.append(issue)
            return issue
        raise AssertionError(f"Unexpected mutation: {path}")


class CommunityTests(unittest.TestCase):
    def test_welcome_and_review_request_are_not_repeated(self):
        github = FakeGitHub()
        for _ in range(3):
            app.community(github, github.pr, ["maintainer"], "")
        self.assertEqual(len(github.records), 1)
        self.assertEqual(sum(path.endswith("requested_reviewers") for _, path, _ in github.writes), 1)
        self.assertIn("@maintainer", github.records[0]["body"])

    def test_draft_is_welcomed_then_routed_when_ready(self):
        github = FakeGitHub(pull(draft=True))
        app.community(github, github.pr, ["maintainer"], "")
        self.assertFalse(github.pr["requested_reviewers"])
        github.pr["draft"] = False
        app.community(github, github.pr, ["maintainer"], "")
        self.assertEqual(len(github.records), 1)
        self.assertEqual(github.pr["requested_reviewers"], [{"login": "maintainer"}])

    def test_author_is_never_requested_and_backup_is_used(self):
        github = FakeGitHub()
        app.route(github, github.pr, ["new-contributor", "backup"])
        self.assertEqual(github.pr["requested_reviewers"], [{"login": "backup"}])

    def test_backup_used_if_primary_cannot_review(self):
        github = FakeGitHub()
        github.fail_reviewers.add("primary")
        app.route(github, github.pr, ["primary", "backup"])
        self.assertEqual(github.pr["requested_reviewers"], [{"login": "backup"}])

    def test_existing_review_or_team_request_is_respected(self):
        for completed in (True, False):
            github = FakeGitHub()
            if completed:
                github.reviews = [{"user": {"login": "maintainer"}}]
            else:
                github.pr["requested_teams"] = [{"slug": "sdk"}]
            app.route(github, github.pr, ["maintainer"])
            self.assertFalse(github.writes)

    def test_bots_and_closed_unmerged_prs_are_skipped(self):
        for pr in (pull(state="closed"), pull(user={"login": "dependabot[bot]", "type": "Bot"})):
            github = FakeGitHub(pr)
            app.community(github, pr, ["maintainer"], "")
            self.assertFalse(github.writes)

    def test_contributor_cannot_spoof_a_welcome_marker(self):
        github = FakeGitHub()
        github.records = [{"id": 1, "body": app.WELCOME, "user": github.pr["user"]}]
        app.community(github, github.pr, ["maintainer"], "")
        self.assertEqual(len(github.records), 2)
        self.assertEqual(github.records[0]["body"], app.WELCOME)

    def test_maintainer_config_is_validated(self):
        self.assertEqual(app.maintainers("Primary, backup Primary"), ["Primary", "backup"])
        with self.assertRaises(ValueError):
            app.maintainers("@everyone")


class CelebrationTests(unittest.TestCase):
    webhook = "https://discord.com/api/webhooks/123/test-token"

    def test_merge_sends_once_and_records_message_id(self):
        github = FakeGitHub(pull(merged=True, state="closed", title="Fix @everyone <@123> [bad](https://evil.test)"))
        with patch.object(app, "request_json", return_value={"id": "987"}) as send:
            app.community(github, github.pr, ["maintainer"], self.webhook)
            app.community(github, github.pr, ["maintainer"], self.webhook)
        send.assert_called_once()
        self.assertEqual(send.call_args.kwargs["data"]["allowed_mentions"], {"parse": []})
        self.assertNotIn("@everyone", send.call_args.kwargs["data"]["content"])
        self.assertIn("?wait=true", send.call_args.args[0])
        self.assertIn("discord:sent:987", github.records[0]["body"])

    def test_missing_secret_can_be_recovered_later(self):
        github = FakeGitHub(pull(merged=True, state="closed"))
        app.celebrate(github, github.pr, "")
        self.assertFalse(github.records)
        with patch.object(app, "request_json", return_value={"id": "987"}) as send:
            app.celebrate(github, github.pr, self.webhook)
        send.assert_called_once()

    def test_timeout_does_not_cause_a_duplicate_on_rerun(self):
        github = FakeGitHub(pull(merged=True, state="closed"))
        with patch.object(app, "request_json", side_effect=app.ServiceError("uncertain")) as send:
            with self.assertRaises(app.ServiceError):
                app.celebrate(github, github.pr, self.webhook)
            app.celebrate(github, github.pr, self.webhook)
        send.assert_called_once()
        self.assertIn("discord:pending", github.records[0]["body"])

    def test_opt_out_supported_in_template_marker_and_label(self):
        for metadata in (
            {"body": "- [x] Please skip the Discord celebration for this PR."},
            {"body": "<!-- cyberwave:no-celebration -->"},
            {"labels": [{"name": "no-celebration"}]},
        ):
            github = FakeGitHub(pull(merged=True, state="closed", **metadata))
            with patch.object(app, "request_json") as send:
                app.celebrate(github, github.pr, self.webhook)
            send.assert_not_called()
            self.assertFalse(github.writes)
        self.assertFalse(app.opted_out(pull(body="- [ ] Please skip the Discord celebration for this PR.")))

    def test_webhook_host_and_redirect_tricks_rejected(self):
        for url in ("http://discord.com/api/webhooks/123/token", "https://evil.test/api/webhooks/123/token",
                    "https://discord.com@evil.test/api/webhooks/123/token",
                    self.webhook + "?redirect=https://evil.test"):
            with self.assertRaises(ValueError):
                app.validate_webhook(url)


class SlackTests(unittest.TestCase):
    webhook = "https://hooks.slack.com/services/T123/B456/test-token"

    def test_open_update_merge_and_replay_send_only_two_messages(self):
        github = FakeGitHub()
        with patch.object(app, "request_json", return_value="ok") as send:
            for action in ("opened", "synchronize", "reopened", "workflow_dispatch"):
                app.notify_slack(github, github.pr, self.webhook, action)
            self.assertEqual(send.call_count, 1)
            self.assertIn("ready for review", send.call_args.kwargs["data"]["text"])
            github.pr.update(state="closed", merged=True)
            app.notify_slack(github, github.pr, self.webhook, "closed")
            app.notify_slack(github, github.pr, self.webhook, "workflow_dispatch")
            self.assertEqual(send.call_count, 2)
            self.assertIn("PR merged", send.call_args.kwargs["data"]["text"])
        self.assertEqual(len(github.records), 1)
        self.assertIn("slack:opened:sent", github.records[0]["body"])
        self.assertIn("slack:merged:sent", github.records[0]["body"])

    def test_draft_gets_one_ready_notification(self):
        github = FakeGitHub(pull(draft=True))
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, self.webhook, "opened")
            self.assertIn("Draft", send.call_args.kwargs["data"]["text"])
            github.pr["draft"] = False
            app.notify_slack(github, github.pr, self.webhook, "ready_for_review")
            app.notify_slack(github, github.pr, self.webhook, "ready_for_review")
            github.pr["draft"] = True
            app.notify_slack(github, github.pr, self.webhook, "reopened")
            self.assertEqual(send.call_count, 2)
        self.assertEqual(len(github.records), 1)

    def test_closed_unmerged_and_stale_ready_events_do_not_notify(self):
        with patch.object(app, "request_json") as send:
            github = FakeGitHub(pull(state="closed"))
            app.notify_slack(github, github.pr, self.webhook, "closed")
            app.notify_slack(github, github.pr, self.webhook, "opened")
            github.pr.update(state="open", draft=True)
            app.notify_slack(github, github.pr, self.webhook, "ready_for_review")
            send.assert_not_called()
        self.assertFalse(github.records)

    def test_delayed_open_event_never_requests_review_of_merged_pr(self):
        github = FakeGitHub(pull(state="closed", merged=True))
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, self.webhook, "opened")
            self.assertIn("PR merged", send.call_args.kwargs["data"]["text"])

    def test_missing_secret_leaves_no_marker_and_can_be_retried(self):
        github = FakeGitHub()
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, "", "opened")
            self.assertFalse(github.records)
            send.assert_not_called()
            app.notify_slack(github, github.pr, self.webhook, "workflow_dispatch")
            send.assert_called_once()

    def test_uncertain_send_is_not_repeated_but_later_merge_can_notify(self):
        github = FakeGitHub()
        with patch.object(app, "request_json", side_effect=app.ServiceError("uncertain")) as send:
            with self.assertRaises(app.ServiceError):
                app.notify_slack(github, github.pr, self.webhook, "opened")
            app.notify_slack(github, github.pr, self.webhook, "workflow_dispatch")
            send.assert_called_once()
        self.assertIn("slack:opened:pending", github.records[0]["body"])
        github.pr.update(state="closed", merged=True)
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, self.webhook, "closed")
            send.assert_called_once()
        self.assertIn("slack:opened:pending", github.records[0]["body"])
        self.assertIn("slack:merged:sent", github.records[0]["body"])

    def test_unexpected_response_is_not_recorded_as_sent(self):
        github = FakeGitHub()
        with patch.object(app, "request_json", return_value="invalid_payload"), self.assertRaises(app.ServiceError):
            app.notify_slack(github, github.pr, self.webhook, "opened")
        self.assertIn("slack:opened:pending", github.records[0]["body"])
        self.assertNotIn("slack:opened:sent", github.records[0]["body"])

    def test_discord_opt_out_does_not_hide_team_review_notifications(self):
        github = FakeGitHub(pull(body="<!-- cyberwave:no-celebration -->"))
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, self.webhook, "opened")
            send.assert_called_once()

    def test_bot_prs_are_included_in_team_review_notifications(self):
        github = FakeGitHub(pull(user={"login": "dependabot[bot]", "type": "Bot"}))
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, self.webhook, "opened")
            send.assert_called_once()

    def test_untrusted_title_is_only_plain_text_and_cannot_override_url(self):
        github = FakeGitHub(pull(title="<!channel> <@U123> <https://evil.test|review>"))
        url = "https://github.com/cyberwave-os/cyberwave-python/pull/42"
        for stage in ("draft", "opened", "merged"):
            with self.subTest(stage=stage):
                payload = app.slack_payload(github, github.pr, stage)
                self.assertNotIn("<!channel>", payload["text"])
                self.assertIn(url, payload["text"])
                sections = [block for block in payload["blocks"] if block["type"] == "section"]
                self.assertEqual(sections[0]["text"]["type"], "plain_text")
                self.assertIn(github.pr["title"], sections[0]["text"]["text"])
                self.assertEqual(sections[1]["text"], {"type": "mrkdwn", "text": f"<{url}>"})
                self.assertEqual(sections[2]["text"]["type"], "plain_text")
                self.assertEqual(payload["blocks"][-1]["elements"][0]["url"], url)

    def test_contributor_cannot_spoof_slack_delivery_record(self):
        github = FakeGitHub()
        github.records = [{"id": 1, "body": app.SLACK + "\n<!-- slack:opened:sent -->", "user": github.pr["user"]}]
        with patch.object(app, "request_json", return_value="ok") as send:
            app.notify_slack(github, github.pr, self.webhook, "opened")
            send.assert_called_once()
        self.assertEqual(len(github.records), 2)

    def test_slack_and_discord_use_independent_delivery_records(self):
        github = FakeGitHub(pull(merged=True, state="closed"))
        with patch.object(app, "request_json", side_effect=[{"id": "987"}, "ok"]) as send:
            app.celebrate(github, github.pr, CelebrationTests.webhook)
            app.notify_slack(github, github.pr, self.webhook, "closed")
            self.assertEqual(send.call_count, 2)
        self.assertEqual(len(github.records), 2)
        self.assertIn("discord:sent:987", github.comment(42, app.MERGE)["body"])
        self.assertIn("slack:merged:sent", github.comment(42, app.SLACK)["body"])

    def test_invalid_webhook_rejected_before_writing_any_state(self):
        github = FakeGitHub()
        for url in ("http://hooks.slack.com/services/T/B/secret", "https://evil.test/services/T/B/secret",
                    "https://hooks.slack.com@evil.test/services/T/B/secret", self.webhook + "?secret=1",
                    "https://hooks.slack.com/anything", self.webhook + "#secret"):
            with self.assertRaises(ValueError):
                app.notify_slack(github, github.pr, url, "opened")
        self.assertFalse(github.writes)
        app.validate_slack_webhook("https://hooks.slack-gov.com/services/T1/B2/secret")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.github = FakeGitHub()
        self.reviewer = Mock(return_value={"summary": "Limited patch review.", "findings": []})

    def run_review(self, key="fake-key"):
        app.advisory_review(self.github, copy.deepcopy(self.github.pr), key, "test-model", self.reviewer)

    def test_same_commit_reviewed_once_and_new_commit_updates_comment(self):
        self.run_review()
        self.run_review()
        self.reviewer.assert_called_once()
        self.github.pr["head"]["sha"] = "b" * 40
        self.run_review()
        self.assertEqual(self.reviewer.call_count, 2)
        self.assertEqual(len(self.github.records), 1)
        self.assertIn("reviewed:" + "b" * 40, self.github.records[0]["body"])
        self.assertFalse(any(path.endswith("/reviews") for _, path, _ in self.github.writes))

    def test_missing_key_is_visible_and_retryable(self):
        self.run_review(key="")
        self.reviewer.assert_not_called()
        self.assertIn("Review unavailable", self.github.records[0]["body"])
        self.assertNotIn("<!-- reviewed:", self.github.records[0]["body"])
        self.run_review()
        self.assertEqual(len(self.github.records), 1)
        self.reviewer.assert_called_once()

    def test_service_failure_is_advisory_and_retryable(self):
        self.reviewer.side_effect = app.ServiceError("failure")
        self.run_review()
        self.assertIn("Review unavailable", self.github.records[0]["body"])
        self.assertNotIn("<!-- reviewed:", self.github.records[0]["body"])

    def test_new_commit_arriving_during_review_discards_result(self):
        def review(*args):
            self.github.pr["head"]["sha"] = "b" * 40
            return {"summary": "Old result", "findings": []}
        self.reviewer.side_effect = review
        self.run_review()
        self.assertFalse(self.github.records)

    def test_closing_or_drafting_during_review_discards_result(self):
        for change in ({"state": "closed"}, {"draft": True}):
            self.github = FakeGitHub()
            def review(*args):
                self.github.pr.update(change)
                return {"summary": "Old result", "findings": []}
            self.reviewer.side_effect = review
            self.run_review()
            self.assertFalse(self.github.records)

    def test_drafts_closed_and_bots_not_sent_to_provider(self):
        for change in ({"draft": True}, {"state": "closed"}, {"user": {"login": "bot", "type": "Bot"}}):
            self.github = FakeGitHub(pull(**change))
            self.run_review()
        self.reviewer.assert_not_called()

    def test_file_budgets_and_generated_files_are_excluded(self):
        files = [file_change("cyberwave/rest/api.py"), file_change("poetry.lock"),
                 file_change("image.png", None), file_change("large.py", "x" * 60001),
                 *[file_change(f"code{i}.py") for i in range(45)]]
        selected, skipped = app.review_files(files)
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(skipped), 9)
        self.github.files = files
        self.run_review()
        self.assertIn("40 of 49", self.github.records[0]["body"])

    def test_no_eligible_patches_skips_provider_without_claiming_clean_review(self):
        self.github.files = [file_change("poetry.lock")]
        self.run_review()
        self.reviewer.assert_not_called()
        self.assertIn("Please review manually", self.github.records[0]["body"])

    def test_claude_payload_uses_custom_policy_is_bounded_and_has_no_tools(self):
        result = {"summary": "A review", "findings": []}
        response = {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(result)}]}
        with patch.object(app, "request_json", return_value=response) as send, patch.object(
                app, "load_review_instructions", return_value="Prioritize SDK resource cleanup."):
            self.assertEqual(app.ask_reviewer([file_change()], "key", "model"), result)
        payload = send.call_args.kwargs["data"]
        self.assertEqual(send.call_args.args[0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(send.call_args.kwargs["anthropic_key"], "key")
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["max_tokens"], 1800)
        self.assertEqual(payload["output_config"]["format"]["type"], "json_schema")
        self.assertIn("Prioritize SDK resource cleanup.", payload["system"])
        self.assertIn("untrusted_patches", payload["messages"][0]["content"])

    def test_policy_is_read_from_trusted_location_and_must_be_nonempty(self):
        self.assertIn("## Robotics behavior", app.load_review_instructions())
        with patch.object(app.Path, "read_text", return_value="") as read, self.assertRaises(app.ServiceError):
            app.load_review_instructions()
        read.assert_called_once_with(encoding="utf-8")
        with patch.object(app.Path, "read_text", side_effect=OSError), self.assertRaises(app.ServiceError):
            app.load_review_instructions()

    def test_refused_incomplete_or_fabricated_file_results_are_rejected(self):
        responses = [
            {"stop_reason": "max_tokens"},
            {"stop_reason": "refusal"},
            {"stop_reason": "end_turn", "content": [{"type": "text", "text":
                json.dumps({"summary": "fake", "findings": [{"path": "not-in-diff.py", "severity": "high", "explanation": "bad"}]})}]},
            {"stop_reason": "end_turn", "content": None},
            None,
        ]
        for response in responses:
            with patch.object(app, "request_json", return_value=response), self.assertRaises(app.ServiceError):
                app.ask_reviewer([file_change()], "key", "model")


class TransportTests(unittest.TestCase):
    def test_slack_acknowledgment_is_plain_text_and_has_no_auth_header(self):
        with patch.object(app, "build_opener") as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value = b'ok'
            response = app.request_json(SlackTests.webhook, method="POST", data={"text": "test"}, text_response=True)
        self.assertEqual(response, "ok")
        headers = dict(factory.return_value.open.call_args.args[0].header_items())
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("X-api-key", headers)

    def test_anthropic_auth_headers_are_not_github_bearer_headers(self):
        with patch.object(app, "build_opener") as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value = b'{}'
            app.request_json("https://api.anthropic.com/v1/messages", method="POST", data={}, anthropic_key="test-key")
        headers = dict(factory.return_value.open.call_args.args[0].header_items())
        self.assertEqual(headers["X-api-key"], "test-key")
        self.assertEqual(headers["Anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", headers)

    def test_write_timeout_not_retried_or_logged_with_url(self):
        with patch.object(app, "build_opener") as factory:
            factory.return_value.open.side_effect = URLError("secret-webhook-token")
            with self.assertRaises(app.ServiceError) as raised:
                app.request_json("https://discord.com/api/webhooks/123/secret", method="POST", data={})
            self.assertNotIn("secret", str(raised.exception))
            self.assertEqual(factory.return_value.open.call_count, 1)

    def test_read_retries_transient_errors(self):
        with patch.object(app, "build_opener") as factory, patch.object(app.time, "sleep"):
            factory.return_value.open.side_effect = URLError("temporary")
            with self.assertRaises(app.ServiceError):
                app.request_json("https://api.github.com/repos/a/b/pulls/42")
            self.assertEqual(factory.return_value.open.call_count, 3)

    def test_redirects_are_refused(self):
        self.assertIsNone(app.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test"))

    def test_github_comments_paginate_past_first_hundred(self):
        github = app.GitHub("a/b", "token")
        batch = [{"id": i, "body": "hello", "user": {"login": "human", "type": "User"}} for i in range(100)]
        target = {"id": 101, "body": app.WELCOME, "user": {"login": app.BOT, "type": "Bot"}}
        with patch.object(github, "api", side_effect=[batch, [target]]) as api:
            self.assertEqual(github.comment(42, app.WELCOME), target)
        self.assertIn("page=2", api.call_args.args[0])


class StarterTests(unittest.TestCase):
    def test_seed_is_idempotent_even_after_issue_closed(self):
        github = FakeGitHub()
        issues = starter.load_issues()
        first = starter.seed(github, issues, "khushisharma22")
        github.issues[0]["state"] = "closed"
        second = starter.seed(github, issues, "khushisharma22")
        self.assertEqual(first, second)
        self.assertEqual(len(github.issues), len(issues))
        self.assertTrue(all("@khushisharma22" in issue["body"] for issue in github.issues))

    def test_seed_rejects_ineligible_mentor_before_writing(self):
        github = FakeGitHub()
        with patch.object(github, "api", return_value={"permission": "read"}), self.assertRaises(ValueError):
            starter.seed(github, starter.load_issues(), "viewer")
        self.assertFalse(github.writes)


class EventTests(unittest.TestCase):
    def test_untrusted_pull_request_event_is_rejected(self):
        with patch.dict(app.os.environ, {"GITHUB_EVENT_NAME": "pull_request"}), self.assertRaises(ValueError):
            app.main()

    def test_manual_dispatch_from_nondefault_branch_is_rejected(self):
        event = {"repository": {"default_branch": "main"}}
        env = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_EVENT_PATH": "ignored",
               "GITHUB_REF": "refs/heads/unreviewed", "GITHUB_REPOSITORY": "cyberwave-os/cyberwave-python"}
        with patch.dict(app.os.environ, env), patch.object(app.Path, "read_text", return_value=json.dumps(event)):
            with self.assertRaisesRegex(ValueError, "default branch"):
                app.main()


if __name__ == "__main__":
    unittest.main()
