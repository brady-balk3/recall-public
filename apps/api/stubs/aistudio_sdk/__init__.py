# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Stand-in for the real ``aistudio_sdk``, which cannot be redistributed.

The upstream package declares no license -- no LICENSE file, and UNKNOWN for
both License and Home-page in its metadata -- so shipping it in a build is not
something we can lawfully do. It arrives transitively
(paddleocr -> paddlex -> aistudio_sdk) and paddlex imports exactly two symbols
from it, in one module: ``errors.NotExistError`` and
``snapshot_download.snapshot_download``. Both belong to the code path that
downloads models from Baidu's cloud.

Recall bundles PP-OCRv6 locally and the packaging spec fails the build when
those weights are missing, so that path is never taken. This module satisfies
the import and nothing else; calling the download raises rather than silently
reaching the network from an application that promises it makes no outbound
calls.
"""
