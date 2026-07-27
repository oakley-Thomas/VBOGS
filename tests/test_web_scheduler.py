import asyncio

from vbogs.web.scheduler import Scheduler
from vbogs.web.store import RunStore, utc_now


def record(tmp_path, run_id):
    return {
        "id": run_id, "owner": "operator", "dataset": "kitti360", "scene_id": "scene",
        "preset": "kitti360-dev", "start_at": "prepare", "stop_after": "bundle",
        "created_at": utc_now(), "config_path": str(tmp_path / f"{run_id}.yaml"),
        "workspace_path": str(tmp_path / run_id), "output_path": str(tmp_path / "outputs" / run_id),
        "command": ["scripts/run_pipeline.sh"],
    }


def test_scheduler_assigns_one_run_per_gpu_and_releases_slot(tmp_path):
    async def exercise():
        store = RunStore(tmp_path / "control.sqlite3")
        store.create_run(record(tmp_path, "run-111111111111"))
        store.create_run(record(tmp_path, "run-222222222222"))
        started = asyncio.Event()
        release = asyncio.Event()

        async def runner(run, gpu):
            started.set()
            await release.wait()
            return 0

        scheduler = Scheduler(store, ("0",), runner)
        scheduler._dispatch()
        await started.wait()
        assert len(store.active_runs()) == 1
        assert len(store.queued_runs()) == 1
        release.set()
        await asyncio.gather(*scheduler.tasks.values())
        scheduler._dispatch()
        await asyncio.sleep(0)
        assert store.get_run("run-111111111111")["status"] == "completed"
        assert store.get_run("run-222222222222")["status"] in {"running", "completed"}
        await asyncio.gather(*scheduler.tasks.values())

    asyncio.run(exercise())
