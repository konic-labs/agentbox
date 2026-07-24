from pathlib import Path

from agentbox.jobs import GenerateJob, LocalArtifactStore, job_from_dict


def test_job_from_dict() -> None:
    job = job_from_dict(
        {
            "job_id": "g1",
            "kind": "generate",
            "model": "m",
            "target": 5,
            "out_dir": "out/tasks",
        }
    )
    assert isinstance(job, GenerateJob)
    assert job.target == 5
    assert job.model == "m"


def test_artifact_run_dir(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    d = store.run_dir("run-abc")
    assert d.exists()
    assert d.name == "run-abc"
