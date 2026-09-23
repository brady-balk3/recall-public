# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""The download entry point paddlex imports. It must never run."""


def snapshot_download(*args, **kwargs):
    raise RuntimeError(
        "Recall does not download models from Baidu AI Studio. OCR weights are "
        "bundled under models/ocr; if this was reached, the bundled PP-OCRv6 "
        "models are missing or PaddleOCR was pointed somewhere else."
    )
