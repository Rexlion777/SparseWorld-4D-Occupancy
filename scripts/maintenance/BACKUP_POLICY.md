# Key Backup Policy

Destination root:

`/mnt/e/ComputerVision_KeyBackup`

Daily backup includes:

- Source/config/test files from project code directories.
- Report and project text/table files up to 10 MiB: `md`, `json`, `csv`, `txt`, `html`.
- Report and project figures up to 1 MiB: `png`, `jpg`, `jpeg`, `svg`.
- Finished videos from `videos`: `mp4`, `webm`, `mov`, `mkv`, `avi`, capped at 200 MiB per file.

Daily backup excludes:

- Model and tensor intermediates: `pt`, `pth`, `ckpt`, `npz`, `npy`, `pkl`, `onnx`, `engine`, `bin`.
- Large generated tables above 10 MiB.
- Frame dumps and large figures above 1 MiB.
- Datasets, external dependencies, logs, outputs, and artifacts.

The `sparseworld_cu128` environment is backed up raw only when running:

```bash
scripts/maintenance/backup_key_materials_to_e.sh --env-only
```

Daily scheduled backup should run:

```bash
scripts/maintenance/backup_key_materials_to_e.sh --daily-only
```

GitHub is for source backup only. Generated reports, videos, artifacts, datasets,
checkpoints, and the local CUDA environment are intentionally ignored by git.

