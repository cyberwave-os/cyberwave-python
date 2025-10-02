from __future__ import annotations

from typing import Any, Dict, List, Optional

from .async_http import AsyncHttpClient


class MLAPI:
    """High-level ML helpers: datasets, training jobs, model instances."""

    def __init__(self, http: AsyncHttpClient):
        self._h = http

    # Datasets
    async def list_datasets(self) -> List[Dict[str, Any]]:
        return await self._h.get("ml/datasets")

    async def create_dataset(self, *, name: str, slug: Optional[str] = None, description: str = "", tags: Optional[list[str]] = None) -> Dict[str, Any]:
        payload = {"name": name, "slug": slug, "description": description, "tags": tags or []}
        return await self._h.post("ml/datasets", payload)

    async def list_dataset_versions(self, dataset_uuid: str) -> List[Dict[str, Any]]:
        return await self._h.get(f"ml/datasets/{dataset_uuid}/versions")

    async def build_dataset_version(
        self,
        dataset_uuid: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        labeling_hook: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {"filters": filters or {}, "labeling_hook": labeling_hook}
        return await self._h.post(f"ml/datasets/{dataset_uuid}/versions", payload)

    async def export_dataset_version(self, dataset_uuid: str, version: int, *, format: str = "parquet") -> Dict[str, Any]:
        payload = {"format": format}
        return await self._h.post(f"ml/datasets/{dataset_uuid}/versions/{version}/export", payload)

    # Training
    async def submit_training_job(self, dataset_uuid: str, version: int, *, trainer: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"trainer": trainer, "hyperparameters": hyperparameters or {}}
        return await self._h.post(f"ml/datasets/{dataset_uuid}/versions/{version}/train", payload)

    async def list_training_jobs(self, *, status: Optional[str] = None, dataset_uuid: Optional[str] = None, version: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if status: params["status"] = status
        if dataset_uuid: params["dataset_uuid"] = dataset_uuid
        if version is not None: params["version"] = version
        return await self._h.get("ml/training/jobs", params=params)

    async def get_training_job(self, job_uuid: str) -> Dict[str, Any]:
        return await self._h.get(f"ml/training/jobs/{job_uuid}")

    # Models & instances
    async def list_models(self) -> List[Dict[str, Any]]:
        return await self._h.get("ml/models")

    async def list_model_instances(self) -> List[Dict[str, Any]]:
        instances = await self._h.get("ml/model-instances")
        try:
            models = await self.list_models()
            model_map = {model["uuid"]: model for model in models}
        except Exception:
            model_map = {}

        for instance in instances:
            model = model_map.get(instance.get("model_uuid"))
            if model and "task" not in instance:
                instance["task"] = model.get("default_task", "custom")
            if "framework" not in instance:
                instance_metadata = instance.get("metadata") or {}
                instance["framework"] = instance_metadata.get("framework", "")
            if "version" not in instance:
                instance_metadata = instance.get("metadata") or {}
                instance["version"] = instance_metadata.get("version_label", "")
        return instances

    # Backwards-compatible aliases
    list_model_runtimes = list_model_instances
    list_model_endpoints = list_model_instances

    async def create_model_instance(
        self,
        *,
        name: str,
        task: str,
        slug: Optional[str] = None,
        description: str = "",
        framework: str = "",
        version: str = "",
        lifecycle_stage: Optional[str] = None,
        deployment_target: Optional[str] = None,
        endpoint_url: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        runtime_identifier: str = "",
    ) -> Dict[str, Any]:
        metadata_payload = dict(metadata or {})
        if framework:
            metadata_payload.setdefault("framework", framework)
        if version:
            metadata_payload.setdefault("version_label", version)

        model = await self._h.post(
            "ml/models",
            {
                "name": name,
                "slug": slug,
                "description": description,
                "default_task": task or "custom",
                "metadata": metadata_payload,
            },
        )

        parsed_version: Optional[int] = None
        if version:
            try:
                parsed_version = int(version)
            except (TypeError, ValueError):
                parsed_version = None

        model_version = await self._h.post(
            f"ml/models/{model['uuid']}/versions",
            {
                "version": parsed_version,
                "trainer": framework,
                "params": metadata_payload,
            },
        )

        instance = await self._h.post(
            "ml/model-instances",
            {
                "model_version_uuid": model_version["uuid"],
                "name": name,
                "slug": slug,
                "description": description,
                "lifecycle_stage": lifecycle_stage or "draft",
                "runtime_target": deployment_target or "cloud",
                "runtime_identifier": runtime_identifier,
                "endpoint_url": endpoint_url,
                "config": config or {},
                "metadata": metadata_payload,
            },
        )

        # Back-compat for callers expecting legacy fields
        instance.setdefault("task", model.get("default_task", task))
        instance.setdefault("framework", framework or metadata_payload.get("framework", ""))
        if parsed_version is not None:
            instance.setdefault("version", str(parsed_version))
        else:
            instance.setdefault("version", version)
        if "deployment_target" not in instance and "runtime_target" in instance:
            instance["deployment_target"] = instance["runtime_target"]

        return instance

    # Backwards-compatible aliases
    create_model_runtime = create_model_instance
    create_model_endpoint = create_model_instance

    async def deploy_model_instance(self, instance_uuid: str) -> Dict[str, Any]:
        return await self._h.post(f"ml/model-instances/{instance_uuid}/deploy", {})

    async def stop_model_instance(self, instance_uuid: str) -> Dict[str, Any]:
        return await self._h.post(f"ml/model-instances/{instance_uuid}/stop", {})

    async def model_instance_health(self, instance_uuid: str) -> Dict[str, Any]:
        return await self._h.get(f"ml/model-instances/{instance_uuid}/health")

    async def promote_model_instance(self, instance_uuid: str, *, stage: str) -> Dict[str, Any]:
        return await self._h.post(f"ml/model-instances/{instance_uuid}/promote", {"stage": stage})

    # Backwards-compatible aliases
    deploy_model_runtime = deploy_model_instance
    stop_model_runtime = stop_model_instance
    model_runtime_health = model_instance_health
    promote_model_runtime = promote_model_instance
    deploy_endpoint = deploy_model_instance
    stop_endpoint = stop_model_instance
    health_endpoint = model_instance_health
    promote_endpoint = promote_model_instance
