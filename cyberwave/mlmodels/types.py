"""Typed return values for :mod:`cyberwave.mlmodels`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MLModelRunResult", "MLModelSummary"]


@dataclass
class MLModelSummary:
    """Lightweight view of a catalog entry returned by ``.get()``.

    Only fields the SDK consumes are included; the full backend schema has
    many more. Use :attr:`raw` when you need them.
    """

    uuid: str
    slug: str | None
    name: str
    model_external_id: str
    model_provider_name: str
    output_format: str | None
    deployment: str
    can_take_image_as_input: bool
    can_take_text_as_input: bool
    playground_kind: str | None = None
    output_family: str | None = None
    allowed_structured_tasks: list[str] = field(default_factory=list)
    execution_surfaces: list[str] = field(default_factory=list)
    sdk_load_id: str | None = None
    edge_catalog_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "MLModelSummary":
        return cls(
            uuid=data["uuid"],
            slug=data.get("slug"),
            name=data.get("name", ""),
            model_external_id=data.get("model_external_id", ""),
            model_provider_name=data.get("model_provider_name", ""),
            output_format=data.get("output_format"),
            deployment=data.get("deployment", "cloud"),
            can_take_image_as_input=bool(data.get("can_take_image_as_input")),
            can_take_text_as_input=bool(data.get("can_take_text_as_input", True)),
            playground_kind=data.get("playground_kind"),
            output_family=data.get("output_family"),
            allowed_structured_tasks=list(data.get("allowed_structured_tasks") or []),
            execution_surfaces=list(data.get("execution_surfaces") or []),
            sdk_load_id=data.get("sdk_load_id"),
            edge_catalog_id=data.get("edge_catalog_id"),
            metadata=dict(data.get("metadata") or {}),
            tags=list(data.get("tags") or []),
            raw=dict(data),
        )


@dataclass
class MLModelRunResult:
    """Result of :meth:`cyberwave.mlmodels.MLModelsClient.run`.

    Mirrors the backend ``MLModelRunResultSchema`` (sync 200 response) and
    carries enough context to visualize or further process the output.

    Use :meth:`is_queued` to branch on async cloud-node workloads.
    """

    status: str  # "completed" | "queued"
    output_format: str | None  # "text"|"json"|"points"|"boxes"|"masks"|"mesh"|"image"|"raw"
    output: Any
    raw: str | None = None

    # Present when ``status == "queued"``: poll the cloud-node workload.
    workload_uuid: str | None = None
    poll_url: str | None = None

    # Context carried for the caller's convenience (e.g. when saving an
    # annotated image we can stamp the model the result came from).
    model_uuid: str | None = None
    model_slug: str | None = None
    structured_task: str | None = None

    def is_queued(self) -> bool:
        """True iff the backend deferred execution to a cloud-node workload."""
        return self.status == "queued"

    def is_completed(self) -> bool:
        return self.status == "completed"

    # ------------------------------------------------------------------
    # Convenience exporters
    # ------------------------------------------------------------------

    def save_annotated_image(
        self,
        source: Any,
        path: str,
        *,
        render: bool = True,
        embed_metadata: bool = True,
    ) -> str:
        """Save ``source`` to ``path`` with overlays and/or embedded metadata.

        Thin wrapper around :func:`cyberwave.image.save_annotated_image`.

        Args:
            source: The image the model was run against (path / bytes / PIL).
            path: Destination PNG path.
            render: When ``True`` (default) draw overlays. When ``False``
                just embed the metadata — faster, lower power, and works for
                text / free-prompt results. See
                :func:`cyberwave.image.save_annotated_image` for details.
            embed_metadata: Whether to store the raw JSON in the PNG
                ``tEXt`` chunk (recoverable via
                :func:`cyberwave.image.read_annotated_metadata`).

        Raises:
            RuntimeError: when called on a queued result, or when
                ``render=True`` but ``output_format`` is not spatial.
        """
        if not self.is_completed():
            raise RuntimeError(
                f"Cannot annotate image for a {self.status!r} result. "
                f"Poll {self.poll_url} until completion first."
            )
        # Only guard when actually drawing — render=False can archive any
        # output_format (including 'text' / 'free') by design.
        if render and self.output_format not in {"points", "boxes", "masks"}:
            raise RuntimeError(
                f"output_format={self.output_format!r} is not a spatial output. "
                f"Pass render=False to archive the result without drawing, "
                f"or call save_annotated_image only for points/boxes/masks."
            )

        # Deferred import so the SDK stays usable without Pillow installed.
        from cyberwave.image import save_annotated_image

        payload = {
            "output_format": self.output_format,
            "output": self.output,
            "raw": self.raw,
            "status": self.status,
            "workload_uuid": self.workload_uuid,
            "model_uuid": self.model_uuid,
        }
        extra = {
            "model_slug": self.model_slug,
            "structured_task": self.structured_task,
        }
        result_path = save_annotated_image(
            source,
            payload,
            path,
            render=render,
            embed_metadata=embed_metadata,
            metadata_extra={k: v for k, v in extra.items() if v is not None},
        )
        return str(result_path)

    @classmethod
    def from_api(cls, data: dict[str, Any], *, status_code: int) -> "MLModelRunResult":
        """Build a result from the raw HTTP response body."""
        if status_code == 202 or data.get("status") == "queued":
            return cls(
                status="queued",
                output_format=None,
                output=None,
                workload_uuid=data.get("workload_uuid"),
                poll_url=data.get("poll_url"),
            )
        return cls(
            status=data.get("status", "completed"),
            output_format=data.get("output_format"),
            output=data.get("output"),
            raw=data.get("raw"),
        )
