# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""The one exception type paddlex imports from aistudio_sdk."""


class NotExistError(Exception):
    """Raised upstream when a remote repo is missing. Never raised here."""
