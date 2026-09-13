# CALVIN HiCoRA

CALVIN-native ActionCodec, full BAR, and HiCoRA pipeline.

**Agent execution plan:** [`AGENT_PLAN.md`](AGENT_PLAN.md) — нормативный
пошаговый план, блокеры, команды и Definition of Done.

The frozen decisions are in `protocols/codec_protocol.json`. See
`CALVIN_PORT.md` for scope and provenance requirements and
`FINDINGS_CALVIN.md` for deviations and results.

## Intended execution order

```bash
# CPU / debug data
bash scripts/fetch_data.sh debug /path/to/calvin_dataset
python scripts/convert_calvin.py \
  --dataset-root /path/to/calvin_dataset/calvin_debug_dataset \
  --output-root data/converted/debug \
  --pipeline-validation
python scripts/selftest_codec.py

# One V100 / full D data
bash scripts/fetch_data.sh D /datasets/calvin
python scripts/convert_calvin.py \
  --dataset-root /datasets/calvin/task_D_D \
  --output-root /datasets/calvin_converted/task_D_D
```

Training and evaluation commands are added only after their CPU self-tests
pass.
