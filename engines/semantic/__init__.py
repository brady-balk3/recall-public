# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Local semantic judge (HUMAN_CLIPS plan, Package C).

The reaction engine measures arousal — how loud a moment is — not meaning.
This package asks a small local instruct LLM (llama.cpp, GGUF) whether each
candidate clip's transcript window actually reads like a moment: does it hook,
is it self-contained, does something land. Fully offline; the judge is an
optional channel in the chat/burst mold — absent model, absent transcript, or
any failure means the engine behaves exactly as it does today.
"""
