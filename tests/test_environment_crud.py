import asyncio
import logging
from cyberwave import Client

logging.basicConfig(level=logging.INFO)

async def main():
    client = Client(use_token_cache=False)
    # assumes backend issues default project if none exist; otherwise, adapt to your env
    projects = await client.get_projects(workspace_id=1)
    project_uuid = str(projects[0]["uuid"]) if projects else ""
    env = await client.create_environment(project_uuid=project_uuid, name="SDK Env Test", description="Test")
    envs = await client.get_environments(project_uuid=project_uuid)
    fetched = await client.get_environment(environment_uuid=str(env.get("uuid")))
    assert any(str(e.get("uuid")) == str(env.get("uuid")) for e in envs)
    assert str(fetched.get("uuid")) == str(env.get("uuid"))
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())


