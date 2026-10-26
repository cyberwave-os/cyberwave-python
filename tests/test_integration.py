"""
Integration test that simulates a typical user workflow with the SDK.

This test demonstrates the complete SDK workflow by making real API calls:
1. Import the SDK
2. Create a project
3. Create an environment
4. Create an asset
5. Create a twin
6. Move the twin
7. Rotate the twin
8. Delete the twin
9. Delete the asset
10. Delete the environment
11. Delete the project

Prerequisites:
- User must be authenticated (set CYBERWAVE_API_KEY env var)
- Get your API key from https://cyberwave.com/profile

To run this test:
    cd cyberwave-sdks/cyberwave-python
    export CYBERWAVE_API_KEY="your_api_key_here"
    poetry install
    poetry run pytest tests/test_integration.py -v -s
"""

import os
import pytest
import time

# Import SDK components
from cyberwave import Cyberwave
from cyberwave.exceptions import CyberwaveAPIError, CyberwaveInsufficientCreditsError


def _delete_resource_best_effort(kind: str, resource_uuid: str, delete_fn) -> None:
    """Best-effort teardown of a resource created by a regression test.

    ``delete_fn`` failures used to be swallowed with a bare ``except Exception:
    pass``, which is why regression-test artifacts (e.g. three "Regression Test
    Asset" rows) survived nightly CI runs and became permanent catalog entries —
    the cleanup call was failing silently with nothing in the CI log to catch it.
    Printing the failure at least makes it discoverable in the nightly run's
    output instead of leaving the row as an invisible leak.
    """
    try:
        delete_fn()
    except Exception as exc:
        print(f"  WARNING: failed to clean up {kind} {resource_uuid}: {exc}")


@pytest.fixture(scope="module")
def cyberwave_client():
    """
    Create a Cyberwave client connected to the Cyberwave API.

    Requires environment variable:
    - CYBERWAVE_API_KEY: Your API key from https://cyberwave.com/profile
    """
    api_key = os.getenv("CYBERWAVE_API_KEY")

    if not api_key:
        pytest.skip(
            "CYBERWAVE_API_KEY environment variable not set. Get your API key from https://cyberwave.com/profile"
        )

    client = Cyberwave(api_key=api_key)

    # Test connection
    try:
        client.twins.list()
    except Exception as e:
        pytest.skip(f"Cannot connect to Cyberwave API. Please check your API key: {e}")

    yield client

    # Cleanup
    client.disconnect()


@pytest.fixture(scope="module")
def cyberwave_client_no_credits():
    """
    Create a Cyberwave client for a test account that has no available credits.

    Used to verify that credit-exhausted responses surface as
    ``CyberwaveInsufficientCreditsError`` rather than a generic crash.

    Requires environment variable:
    - CYBERWAVE_API_KEY_NO_CREDITS: API key for an account whose credits have
      been drained or whose organization has been force-blocked by CI setup steps.
    """
    api_key = os.getenv("CYBERWAVE_API_KEY_NO_CREDITS")

    if not api_key:
        pytest.skip(
            "CYBERWAVE_API_KEY_NO_CREDITS not set — skipping no-credits tests. "
            "Set this to the API key of an account with zero or negative credits."
        )

    client = Cyberwave(api_key=api_key)

    # Read-only sanity check — listing twins does not charge credits.
    try:
        client.twins.list()
    except CyberwaveInsufficientCreditsError:
        # Account is already blocked even for reads — that's fine for this fixture.
        pass
    except Exception as e:
        pytest.skip(f"Cannot connect with no-credits API key: {e}")

    yield client

    client.disconnect()


class TestIntegrationWorkflow:
    """
    Integration test that simulates a complete user workflow with real API calls.

    This test validates that all SDK components work together correctly
    to perform typical operations a user would do.
    """

    def test_complete_user_workflow(self, cyberwave_client):
        """
        Test the complete workflow from project creation to cleanup.

        This test makes real API calls to perform:
        1. Create a project
        2. Create an environment
        3. Create an asset
        4. Create a twin from the asset
        5. Move the twin to a new position
        6. Rotate the twin
        7. Clean up all created resources
        """
        client = cyberwave_client

        # Store created resources for cleanup
        created_resources = {
            "project": None,
            "environment": None,
            "asset": None,
            "twin": None,
        }

        try:
            # Step 1: Create a project
            print("\n" + "=" * 70)
            print("Starting Integration Test: Complete User Workflow")
            print("=" * 70)

            project = client.projects.create(
                name=f"SDK Integration Test {int(time.time())}",
                workspace_id=None,
                description="Auto-generated project for SDK integration testing",
            )
            created_resources["project"] = project

            assert project.uuid is not None
            assert project.name.startswith("SDK Integration Test")
            print(f"✓ Step 1: Created project '{project.name}' (UUID: {project.uuid})")

            # Step 2: Create an environment in the project
            environment = client.environments.create(
                name=f"Test Environment {int(time.time())}",
                project_id=project.uuid,
                description="Auto-generated environment for SDK testing",
            )
            created_resources["environment"] = environment

            assert environment.uuid is not None
            assert environment.project_uuid == project.uuid
            print(
                f"✓ Step 2: Created environment '{environment.name}' (UUID: {environment.uuid})"
            )

            # Step 3: Create an asset
            asset = client.assets.create(
                name=f"Test Asset {int(time.time())}",
                description="Auto-generated asset for SDK integration testing",
                asset_type="generic",
            )
            created_resources["asset"] = asset

            assert asset.uuid is not None
            print(f"✓ Step 3: Created asset '{asset.name}' (UUID: {asset.uuid})")

            # Step 4: Create a twin from the asset
            twin_data = client.twins.create(
                asset_id=asset.uuid, environment_id=environment.uuid
            )
            created_resources["twin"] = twin_data

            assert twin_data.uuid is not None
            assert twin_data.asset_uuid == asset.uuid
            assert twin_data.environment_uuid == environment.uuid
            print(f"✓ Step 4: Created twin (UUID: {twin_data.uuid})")

            # Step 5: Update twin position using lower-level API
            # (Avoiding Twin class abstraction to test core API functionality)
            updated_twin = client.twins.update_state(
                twin_data.uuid,
                {"position_x": 1.0, "position_y": 0.5, "position_z": 0.3},
            )
            assert updated_twin.position_x == 1.0
            assert updated_twin.position_y == 0.5
            assert updated_twin.position_z == 0.3
            print(
                f"✓ Step 5: Moved twin to position ({updated_twin.position_x}, {updated_twin.position_y}, {updated_twin.position_z})"
            )

            # Step 6: Update twin rotation using lower-level API
            import math

            from cyberwave._geometry import core

            # Convert 45 degree yaw to quaternion (rotation around Z axis).
            # Built by the shared geometry core rather than by hand: a test that
            # spells the half-angle formula out itself is a second
            # implementation, and would keep passing if the real one drifted.
            rotation = core().quat.from_yaw(math.radians(45))
            updated_twin = client.twins.update_state(
                twin_data.uuid,
                {
                    "rotation_w": rotation.w,
                    "rotation_x": rotation.x,
                    "rotation_y": rotation.y,
                    "rotation_z": rotation.z,
                },
            )
            print("✓ Step 6: Rotated twin by 45 degrees (yaw)")

        finally:
            # Cleanup: Delete resources in reverse order
            print("\n" + "-" * 70)
            print("Cleanup Phase")
            print("-" * 70)

            # Step 7: Delete the twin
            if created_resources["twin"]:
                try:
                    client.twins.delete(created_resources["twin"].uuid)
                    print(
                        f"✓ Step 7: Deleted twin (UUID: {created_resources['twin'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete twin: {e}")

            # Step 8: Delete the asset
            if created_resources["asset"]:
                try:
                    client.assets.delete(created_resources["asset"].uuid)
                    print(
                        f"✓ Step 8: Deleted asset (UUID: {created_resources['asset'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete asset: {e}")

            # Step 9: Delete the environment
            if created_resources["environment"] and created_resources["project"]:
                try:
                    client.environments.delete(
                        created_resources["environment"].uuid,
                        created_resources["project"].uuid,
                    )
                    print(
                        f"✓ Step 9: Deleted environment (UUID: {created_resources['environment'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete environment: {e}")

            # Step 10: Delete the project
            if created_resources["project"]:
                try:
                    client.projects.delete(created_resources["project"].uuid)
                    print(
                        f"✓ Step 10: Deleted project (UUID: {created_resources['project'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete project: {e}")

            print("\n" + "=" * 70)
            print("✅ Integration Test Completed Successfully!")
            print("=" * 70 + "\n")


class TestIntegrationErrorHandling:
    """Test error handling in the integration workflow with real API"""

    def test_invalid_twin_creation(self, cyberwave_client):
        """Test handling when twin creation fails with invalid data"""
        print("\n" + "=" * 70)
        print("Testing Error Handling: Invalid Twin Creation")
        print("=" * 70)

        # Try to create a twin with invalid asset and environment IDs
        with pytest.raises(CyberwaveAPIError) as exc_info:
            cyberwave_client.twins.create(
                asset_id="00000000-0000-0000-0000-000000000000",
                environment_id="00000000-0000-0000-0000-000000000000",
            )

        assert (
            "create twin" in str(exc_info.value).lower()
            or "404" in str(exc_info.value)
            or "not found" in str(exc_info.value).lower()
        )
        print("✓ Invalid twin creation error handled correctly")
        print("=" * 70 + "\n")


class TestIntegrationReadOperations:
    """Test read-only operations that don't modify data"""

    def test_list_all_resources(self, cyberwave_client):
        """Test listing various resources"""
        print("\n" + "=" * 70)
        print("Testing Read Operations")
        print("=" * 70)

        # List projects
        projects = cyberwave_client.projects.list()
        print(f"✓ Listed {len(projects)} project(s)")

        # List environments
        environments = cyberwave_client.environments.list()
        print(f"✓ Listed {len(environments)} environment(s)")

        # List assets
        assets = cyberwave_client.assets.list()
        print(f"✓ Listed {len(assets)} asset(s)")

        # List twins
        twins = cyberwave_client.twins.list()
        print(f"✓ Listed {len(twins)} twin(s)")

        print("=" * 70 + "\n")


class TestIntegrationWorkflows:
    """Test workflow trigger → poll → result flow with real API calls."""

    def test_workflow_list_and_trigger(self, cyberwave_client):
        """
        List workflows, trigger an active one, poll until completion.

        Skips if no active workflow is available.
        """
        print("\n" + "=" * 70)
        print("Testing Workflows: list → trigger → poll")
        print("=" * 70)

        client = cyberwave_client

        # List workflows
        workflows = client.workflows.list()
        print(f"✓ Listed {len(workflows)} workflow(s)")

        # Find an active workflow this account can trigger. Listing is
        # visibility-scoped — it also returns public and org-visible workflows
        # from other tenants — while /trigger requires write access, so a
        # foreign workflow at the head of the list fails with a 403. Keep to the
        # account's own workspaces, as the credits test below does.
        own_workspaces = {ws.uuid for ws in client.workspaces.list()}
        active = [
            wf
            for wf in workflows
            if wf.is_active and wf.workspace_uuid in own_workspaces
        ]
        if not active:
            pytest.skip("No active workflows found — skipping trigger test")

        wf = active[0]
        print(f"  Using workflow: {wf.name} ({wf.uuid})")

        # Get workflow by ID
        fetched = client.workflows.get(wf.uuid)
        assert fetched.uuid == wf.uuid
        print(f"✓ Fetched workflow by UUID")

        # Trigger the workflow
        run = client.workflows.trigger(wf.uuid, inputs={})
        assert run.uuid
        assert run.status in ("running", "requested", "waiting", "success", "error")
        print(f"✓ Triggered run {run.uuid} (status: {run.status})")

        # Poll for completion (short timeout — we just verify the flow works)
        try:
            run.wait(timeout=30, poll_interval=2)
            print(f"✓ Run completed: status={run.status}, result={run.result}")
        except Exception:
            print(f"  Run still in progress after 30s (status: {run.status}) — OK for CI")

        # List runs for this workflow
        runs = client.workflow_runs.list(workflow_id=wf.uuid)
        assert len(runs) >= 1
        print(f"✓ Listed {len(runs)} run(s) for workflow")

        # Fetch the run we created
        fetched_run = client.workflow_runs.get(run.uuid)
        assert fetched_run.uuid == run.uuid
        print(f"✓ Fetched run by UUID")

        # Test convenience methods
        wf_runs = wf.runs()
        assert isinstance(wf_runs, list)
        print(f"✓ workflow.runs() returned {len(wf_runs)} run(s)")

        print("=" * 70 + "\n")


class TestIntegrationRegressions:
    """
    Regression tests for bugs that were not caught before shipping.

    Each test documents a specific production bug and verifies the fix is in place.
    """

    def test_twin_with_slash_registry_id(self, cyberwave_client):
        """
        Regression test for CYB-1720 bug #2.

        Calling ``cw.twin("vendor/model")`` raised ``CyberwaveError: Asset not
        found`` because ``get_by_registry_id`` tried ``GET /assets/vendor/model``
        which Django's URL router split at the slash, returning 404.

        The SDK fix routes slash-containing identifiers through the
        ``?registry_id=`` query parameter instead of the URL path.

        The registry_id prefix must match an existing workspace slug so the
        created asset stays in a workspace visible to the API token (backend
        derives workspace from the ``vendor/`` segment of ``vendor/model``).
        """
        print("\n" + "=" * 70)
        print("Regression: cw.twin() with slash-containing registry_id")
        print("=" * 70)

        client = cyberwave_client
        workspace_slug = None
        workspace_id = client.config.workspace_id or os.getenv("CYBERWAVE_WORKSPACE_ID")
        if workspace_id:
            workspace = client.workspaces.get(workspace_id)
            workspace_slug = getattr(workspace, "slug", None)
        if not workspace_slug:
            workspaces = client.workspaces.list()
            if not workspaces:
                pytest.skip("No workspaces available for regression test")
            workspace_slug = getattr(workspaces[0], "slug", None)
        if not workspace_slug:
            pytest.skip("Workspace has no slug for registry_id prefix")
        registry_id = f"{workspace_slug}/test-robot-{int(time.time())}"

        created_resources: dict = {"asset": None, "environment": None, "project": None}

        try:
            project = client.projects.create(
                name=f"SDK Regression Test {int(time.time())}",
                workspace_id=None,
                description="Auto-generated project for SDK regression testing",
            )
            created_resources["project"] = project

            environment = client.environments.create(
                name=f"Regression Test Env {int(time.time())}",
                project_id=project.uuid,
            )
            created_resources["environment"] = environment

            asset = client.assets.create(
                name=f"Regression Test Asset {int(time.time())}",
                description="Created for slash-registry_id regression test",
                asset_type="generic",
                registry_id=registry_id,
            )
            created_resources["asset"] = asset

            print(
                f"  Created asset with registry_id='{registry_id}' (UUID: {asset.uuid})"
            )

            # This was the failing call — must not raise CyberwaveError
            twin = client.twin(registry_id, environment_id=environment.uuid)

            assert twin is not None, "twin() must return a Twin instance"
            assert twin.uuid is not None, "returned twin must have a valid UUID"
            print(f"✓ cw.twin('{registry_id}') succeeded (twin UUID: {twin.uuid})")

        finally:
            if created_resources["asset"]:
                _delete_resource_best_effort(
                    "asset",
                    created_resources["asset"].uuid,
                    lambda: client.assets.delete(created_resources["asset"].uuid),
                )
            if created_resources["environment"] and created_resources["project"]:
                _delete_resource_best_effort(
                    "environment",
                    created_resources["environment"].uuid,
                    lambda: client.environments.delete(
                        created_resources["environment"].uuid,
                        created_resources["project"].uuid,
                    ),
                )
            if created_resources["project"]:
                _delete_resource_best_effort(
                    "project",
                    created_resources["project"].uuid,
                    lambda: client.projects.delete(created_resources["project"].uuid),
                )

        print("=" * 70 + "\n")

    def test_get_latest_frame_returns_valid_jpeg_bytes(self, cyberwave_client):
        """
        Regression test for CYB-1720 bug #1.

        ``twin.get_latest_frame()`` was returning an invalid / truncated image.
        When the ``mock=True`` flag is passed the backend returns a deterministic
        JPEG, allowing us to verify:

        1. The return value is ``bytes`` (not ``None``, not a string).
        2. The bytes start with the JPEG magic number ``\\xff\\xd8`` and end
           with the EOI marker ``\\xff\\xd9``.
        """
        print("\n" + "=" * 70)
        print("Regression: twin.get_latest_frame() returns valid JPEG bytes")
        print("=" * 70)

        client = cyberwave_client

        created_resources: dict = {"asset": None, "environment": None, "project": None}

        try:
            project = client.projects.create(
                name=f"SDK Frame Regression {int(time.time())}",
                workspace_id=None,
                description="Auto-generated project for frame regression test",
            )
            created_resources["project"] = project

            environment = client.environments.create(
                name=f"Frame Regression Env {int(time.time())}",
                project_id=project.uuid,
            )
            created_resources["environment"] = environment

            asset = client.assets.create(
                name=f"Frame Regression Asset {int(time.time())}",
                description="Created for get_latest_frame regression test",
                asset_type="generic",
            )
            created_resources["asset"] = asset

            twin_data = client.twins.create(
                asset_id=asset.uuid,
                environment_id=environment.uuid,
            )

            # Use mock=True so the server always returns a deterministic JPEG
            # without needing a real camera stream to be active.
            frame_bytes = client.twins.get_latest_frame(twin_data.uuid, mock=True)

            assert isinstance(
                frame_bytes, bytes
            ), f"get_latest_frame must return bytes, got {type(frame_bytes)}"
            assert len(frame_bytes) > 0, "returned frame must not be empty"
            assert frame_bytes[:2] == b"\xff\xd8", (
                "JPEG bytes must start with the SOI marker \\xff\\xd8, "
                f"got {frame_bytes[:2]!r}"
            )
            assert frame_bytes[-2:] == b"\xff\xd9", (
                "JPEG bytes must end with the EOI marker \\xff\\xd9, "
                f"got {frame_bytes[-2:]!r}"
            )

            print(
                f"✓ get_latest_frame(mock=True) returned {len(frame_bytes)} valid JPEG bytes"
            )

            # Also verify the Twin-level wrapper produces the same result
            from cyberwave.twin import create_twin

            twin = create_twin(client, twin_data)
            twin_frame_bytes = twin.get_latest_frame(mock=True)
            assert isinstance(twin_frame_bytes, bytes)
            assert len(twin_frame_bytes) > 0
            assert twin_frame_bytes[:2] == b"\xff\xd8"
            assert twin_frame_bytes[-2:] == b"\xff\xd9"
            print(
                f"✓ Twin.get_latest_frame(mock=True) returned {len(twin_frame_bytes)} valid JPEG bytes"
            )

        finally:
            if created_resources["asset"]:
                _delete_resource_best_effort(
                    "asset",
                    created_resources["asset"].uuid,
                    lambda: client.assets.delete(created_resources["asset"].uuid),
                )
            if created_resources["environment"] and created_resources["project"]:
                _delete_resource_best_effort(
                    "environment",
                    created_resources["environment"].uuid,
                    lambda: client.environments.delete(
                        created_resources["environment"].uuid,
                        created_resources["project"].uuid,
                    ),
                )
            if created_resources["project"]:
                _delete_resource_best_effort(
                    "project",
                    created_resources["project"].uuid,
                    lambda: client.projects.delete(created_resources["project"].uuid),
                )

        print("=" * 70 + "\n")


class TestIntegrationCredits:
    """
    Verify that the SDK surfaces credit-exhausted errors as the typed exception
    ``CyberwaveInsufficientCreditsError`` rather than a generic crash.

    These tests require a second API key (``CYBERWAVE_API_KEY_NO_CREDITS``) that
    belongs to an account whose credits have been drained or force-blocked.
    The CI workflow sets this up automatically before the test run.
    """

    def test_workflow_trigger_raises_insufficient_credits(
        self, cyberwave_client_no_credits
    ):
        """
        Triggering a workflow on a no-credits account must raise
        ``CyberwaveInsufficientCreditsError``, not a generic exception.
        """
        print("\n" + "=" * 70)
        print("Testing Credits: trigger raises CyberwaveInsufficientCreditsError")
        print("=" * 70)

        # The workflow has to be one the no-credits account *owns*. Credit
        # enforcement charges the organization that owns the workflow, not the
        # caller's, so triggering someone else's workflow bills their org and
        # never returns 402 no matter how blocked the caller is. Listing is
        # visibility-scoped rather than ownership-scoped — it also returns
        # public and org-visible workflows from other tenants — so filter down
        # to this account's own workspaces explicitly.
        own_workspaces = {
            ws.uuid for ws in cyberwave_client_no_credits.workspaces.list()
        }
        active = [
            wf
            for wf in cyberwave_client_no_credits.workflows.list()
            if wf.is_active and wf.workspace_uuid in own_workspaces
        ]
        if not active:
            # A skip here would hide the gap indefinitely, and CI has already
            # committed to running this test by providing the key.
            pytest.fail(
                "The no-credits account owns no active workflow, so a 402 on "
                "trigger cannot be provoked. CI provisions one in "
                "`.github/scripts/setup_credit_state.py`; reaching this means "
                "that step did not run or its workflow was removed."
            )

        wf = active[0]
        print(f"  Using workflow: {wf.name} ({wf.uuid})")

        with pytest.raises(CyberwaveInsufficientCreditsError) as exc_info:
            cyberwave_client_no_credits.workflows.trigger(wf.uuid, inputs={})

        err = exc_info.value
        assert err.status_code == 402, f"Expected 402, got {err.status_code}"
        if not err.manual_block:
            # Balance-exhausted path: balance must be at or below zero.
            assert err.balance is not None, "balance should be parseable from response"
            assert err.balance <= 0, f"Expected non-positive balance, got {err.balance}"
        print(
            f"✓ trigger raised CyberwaveInsufficientCreditsError "
            f"(balance={err.balance}, manual_block={err.manual_block})"
        )
        print("=" * 70 + "\n")


# -- MQTT v5 request/response through the live broker --------------------------------
#
# The unit tests mock paho, so none of them can tell whether the broker forwards a
# request's Response Topic and Correlation Data, or whether the ACL lets a reply
# through to its requester and to nobody else. These run the exchange against the
# local backend's real broker with the real Django auth and ACL in the path.
#
# Not covered here: that a client is *refused* another requester's reply topic. CI's
# ``test-api-key`` is the shared admin token, and the ACL lets staff principals
# through before any topic rule, so a denial cannot be provoked with it. The
# deny cases run in the backend's ACL tests (``test_mqtt_broker_auth_acl_security``),
# which use ordinary principals.
#
# Broker: ``CYBERWAVE_E2E_MQTT_HOST`` / ``_PORT`` (default localhost:1883, where
# the local compose stack publishes it) and ``CYBERWAVE_E2E_MQTT_TOPIC_PREFIX``
# (default none). Unreachable broker: skipped locally, a failure in CI.


@pytest.fixture(scope="module")
def mqtt_endpoint():
    import socket

    host = os.getenv("CYBERWAVE_E2E_MQTT_HOST", "localhost")
    port = int(os.getenv("CYBERWAVE_E2E_MQTT_PORT", "1883"))
    try:
        socket.create_connection((host, port), timeout=3).close()
    except OSError as exc:
        message = f"MQTT broker {host}:{port} is not reachable: {exc}"
        if os.getenv("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return host, port, os.getenv("CYBERWAVE_E2E_MQTT_TOPIC_PREFIX", "")


@pytest.fixture(scope="module")
def scratch_twin_uuid(cyberwave_client):
    """A twin the API key owns, so its MQTT principal may read and write it."""
    client = cyberwave_client
    workspaces = client.workspaces.list()
    slug = getattr(workspaces[0], "slug", None) if workspaces else None
    if not slug:
        pytest.skip("No workspace with a slug to register a scratch asset under")
    stamp = int(time.time())
    created: dict = {}
    try:
        created["project"] = client.projects.create(
            name=f"SDK request/response {stamp}",
            workspace_id=None,
            description="scratch",
        )
        created["environment"] = client.environments.create(
            name=f"SDK request/response env {stamp}",
            project_id=created["project"].uuid,
        )
        created["asset"] = client.assets.create(
            name=f"SDK request/response asset {stamp}",
            description="scratch",
            asset_type="generic",
            registry_id=f"{slug}/rr-robot-{stamp}",
        )
        twin = client.twin(
            f"{slug}/rr-robot-{stamp}", environment_id=created["environment"].uuid
        )
        yield str(twin.uuid)
    finally:
        if "asset" in created:
            _delete_resource_best_effort(
                "asset",
                created["asset"].uuid,
                lambda: client.assets.delete(created["asset"].uuid),
            )
        if "environment" in created and "project" in created:
            _delete_resource_best_effort(
                "environment",
                created["environment"].uuid,
                lambda: client.environments.delete(
                    created["environment"].uuid, created["project"].uuid
                ),
            )
        if "project" in created:
            _delete_resource_best_effort(
                "project",
                created["project"].uuid,
                lambda: client.projects.delete(created["project"].uuid),
            )


def _v5_client(endpoint, tag):
    import paho.mqtt.client as paho

    from cyberwave.mqtt import CyberwaveMQTTClient

    host, port, prefix = endpoint
    client = CyberwaveMQTTClient(
        mqtt_broker=host,
        mqtt_port=port,
        api_key=os.environ["CYBERWAVE_API_KEY"],
        protocol=paho.MQTTv5,
        topic_prefix=prefix,
        client_id_prefix=f"sdk_e2e_{tag}_",
    )
    client.connect()
    assert client.connected, f"{tag} client did not connect"
    return client


def _wait_subscribed(client, timeout=10.0):
    """Wait for every SUBACK, then give the broker a moment to route."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with client._subscription_lock:
            if not client._pending_subscriptions:
                break
        time.sleep(0.05)
    time.sleep(0.5)


class TestIntegrationRequestResponse:
    """``request`` / ``request_command`` through a real broker and ACL."""

    def test_request_on_a_topic_pair_that_is_not_a_command(
        self, cyberwave_client, mqtt_endpoint, scratch_twin_uuid
    ):
        from cyberwave.mqtt import pass_message
        from cyberwave.mqtt.request_response import reply_to

        prefix = mqtt_endpoint[2]
        request_topic = f"{prefix}cyberwave/twin/{scratch_twin_uuid}/planner/request"
        requester = _v5_client(mqtt_endpoint, "req")
        responder = _v5_client(mqtt_endpoint, "resp")
        try:

            def answer(data, message):
                reply_to(
                    responder.client,
                    message,
                    {"route": ["a", "b"], "goal": data.get("goal")},
                )

            responder.subscribe(request_topic, pass_message(answer), qos=1)
            _wait_subscribed(responder)

            reply = requester.request(
                request_topic,
                {"goal": "b"},
                response_topic=(
                    f"{prefix}cyberwave/twin/{scratch_twin_uuid}/planner/"
                    f"{requester.client_id}/response"
                ),
                timeout_s=20,
            )

            assert reply == {"route": ["a", "b"], "goal": "b"}
        finally:
            requester.disconnect()
            responder.disconnect()

    def test_request_command_is_answered_with_the_command_contract(
        self, cyberwave_client, mqtt_endpoint, scratch_twin_uuid
    ):
        from cyberwave.mqtt import pass_message
        from cyberwave.mqtt.command_response import (
            CommandOutcome,
            command_topic,
            publish_command_response,
        )

        prefix = mqtt_endpoint[2]
        requester = _v5_client(mqtt_endpoint, "cmdreq")
        driver = _v5_client(mqtt_endpoint, "driver")
        try:

            def answer(data, message):
                publish_command_response(
                    driver.client,
                    message,
                    CommandOutcome.accept(),
                    twin_uuid=scratch_twin_uuid,
                    command=data["command"],
                    topic_prefix=prefix,
                )

            driver.subscribe(
                command_topic(scratch_twin_uuid, topic_prefix=prefix),
                pass_message(answer),
                qos=1,
            )
            _wait_subscribed(driver)

            reply = requester.request_command(scratch_twin_uuid, "arm", timeout_s=20)

            assert reply["status"] == "accepted"
            assert reply["command"] == "arm"
        finally:
            requester.disconnect()
            driver.disconnect()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
