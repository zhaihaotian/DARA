# Original training split

```bash
python scripts/fetch_data.py
```

The script fetches `train.parquet` and `test.parquet` from the public
[ToolRL source](https://github.com/qiancheng0/ToolRL/tree/8cee13ec0ca72f0461da372a93a6fd8140dbb840/dataset/rlla_4k),
pinned to revision `8cee13ec0ca72f0461da372a93a6fd8140dbb840`.
The downloaded files were compared directly with the Haotian experiment files:
both are byte-for-byte identical. They contain 3920 train and 80 validation rows.

The upstream data includes token-shaped strings in example tool parameter defaults.
GitHub push protection rejects embedding this parquet in the new repository.
The fixed original data remains externally downloaded; it is not rewritten or
replaced with a different split. The parquet files are ignored by Git.
