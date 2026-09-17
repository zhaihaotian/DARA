# Original training split

```bash
python scripts/fetch_data.py
```

The script fetches `train.parquet` and `test.parquet` from the public
[ToolRL source](https://github.com/qiancheng0/ToolRL/tree/8cee13ec0ca72f0461da372a93a6fd8140dbb840/dataset/rlla_4k),
pinned to revision `8cee13ec0ca72f0461da372a93a6fd8140dbb840`.
The downloaded files were compared directly with the Haotian experiment files:
both are byte-for-byte identical. They contain 3920 train and 80 validation rows.

The script places the fixed parquet files in this directory. Training uses
these downloaded files, and Git tracks the source reference and download script.
