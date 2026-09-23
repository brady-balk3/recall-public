# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Immutable speech artifacts verified against public upstream metadata.

Sizes and LFS hashes were read from the pinned Hugging Face revisions;
small support-file hashes were computed from those revisions' exact bytes.
These pins establish artifact identity, not model quality or licensing approval.
"""

ASR_REPO = "ggml-org/Qwen3-ASR-1.7B-GGUF"
ASR_REVISION = "36a678687ba7d07a74ca70ccb0e36902e005fb80"
ASR_FILES = {
    "Qwen3-ASR-1.7B-Q8_0.gguf": (2165034944, "58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57"),
    "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf": (355709344, "46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d"),
}
ALIGNER_REPO = "valoomba/Qwen3-ForcedAligner-0.6B-ONNX"
ALIGNER_REVISION = "261c9ed100c1b18a4a1fbc488e05625dc9a4ae5c"
ALIGNER_FILES = {
    "config.json": (5725, "91f38394bd8117ad2ccbdfc0942d9c100a7482d4c43688674ef2a40fe64eb061"),
    "preprocessor_config.json": (357, "0443a40d411d3caad769e7b59971fa6eae75926ed4b7e337e60da909eb38c1f7"),
    "tokenizer.json": (11429733, "e95e4127f5ea82f89695f00bb3c143eb7edd411e6096e27ff56319397f481e77"),
    "tokenizer_config.json": (12696, "bee4786e25db33d3e8acb4c7a37c4ccf8d5a3d843b9cafb75a91d4ba774ad38f"),
    "added_tokens.json": (1591, "85afd03b7a7a9b0a97edf2b0e5dd9b0bf8770e05b0323268c0f1a3e5190cfa6f"),
    "special_tokens_map.json": (1008, "7b376c510ccf9d88bb9bbee41dfc5052122e16e0dec1124a8d8983c59259a9f3"),
    "export_metadata.json": (5355, "c2b366f1c413b7031f56bf0cb14ad76d62e94d277d90f0a5e0330fd1a1b06dd9"),
    "onnx/model.onnx": (1917718, "7b2bff8b7a8df4120b450673d65915488e5fd43e107f87f3816d9e2c830a9e8b"),
    "onnx/model.onnx_data": (3670969192, "429a19b51f5f8a2504b5e573de87c8f74e2ad0be68e0ff7ba1e645663507bcdc"),
    "onnx/model_q4.onnx": (1048397349, "59b528896d70b34e57838e160d16d5f7cfc02d86c7c6ad46cdc57c25c15497b7"),
}

VAD_REPO = "istupakov/silero-vad-onnx"
VAD_REVISION = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"
VAD_FILES = {"silero_vad.onnx": (2327524, "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3")}
SPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
SPEAKER_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
SPEAKER_FILES = {"voxceleb_resnet34_LM.onnx": (26530309, "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068")}
