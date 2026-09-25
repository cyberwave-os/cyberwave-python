"""resolve_inbound_topics for depth and pointcloud streams."""

from cyberwave.manifest.driver_config import (
    TWIN_DEPTH_TOPIC_SLUG,
    TWIN_POINTCLOUD_TOPIC_SLUG,
    resolve_inbound_topics,
)


def test_depth_topic_fallback_when_not_in_catalog() -> None:
    resolved = resolve_inbound_topics("depth", {}, twin_uuid="t1")
    assert resolved == [(TWIN_DEPTH_TOPIC_SLUG, "cyberwave/twin/t1/depth")]


def test_pointcloud_topic_fallback_when_not_in_catalog() -> None:
    resolved = resolve_inbound_topics("pointcloud", None, twin_uuid="t1")
    assert resolved == [
        (TWIN_POINTCLOUD_TOPIC_SLUG, "cyberwave/twin/t1/pointcloud")
    ]


def test_depth_topic_honors_prefix() -> None:
    resolved = resolve_inbound_topics(
        "depth", {}, twin_uuid="t1", topic_prefix="staging/"
    )
    assert resolved == [(TWIN_DEPTH_TOPIC_SLUG, "staging/cyberwave/twin/t1/depth")]


def test_select_listen_slugs_accepts_depth_filter() -> None:
    from cyberwave.manifest.driver_config import (
        TWIN_DEPTH_TOPIC_SLUG,
        select_listen_slugs,
    )

    slugs = select_listen_slugs(
        {"topics": {TWIN_DEPTH_TOPIC_SLUG: {}}}, filters=["depth"]
    )
    assert TWIN_DEPTH_TOPIC_SLUG in slugs


def test_select_listen_slugs_accepts_pointcloud_filter() -> None:
    from cyberwave.manifest.driver_config import (
        TWIN_POINTCLOUD_TOPIC_SLUG,
        select_listen_slugs,
    )

    slugs = select_listen_slugs(
        {"topics": {TWIN_POINTCLOUD_TOPIC_SLUG: {}}}, filters=["pointcloud"]
    )
    assert TWIN_POINTCLOUD_TOPIC_SLUG in slugs
