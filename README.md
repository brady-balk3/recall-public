# Recall

Recall is a local Windows desktop app for finding, reviewing, and exporting short clips from long streams and recordings. Its review tools let you adjust framing, captions, and clip boundaries before export.

## Install

Download the Windows installer from this repository's Releases page when a release is available. The small setup program downloads a checksum-verified application package from GitHub and the pinned large model weights from Hugging Face. You do not need to download model files manually. The first install needs an internet connection and enough disk space for the application, models, and your source videos.

Your recordings, scan data, and exported clips are stored locally. Downloading a Twitch VOD, application package, or model uses the network.

## Build from source

The application source lives in `apps/`, `core/`, `engines/`, and `pipeline/`. The desktop uses Electron and React; the local engine uses Python. Model weights, downloaded runtimes, recordings, and generated builds are excluded from Git.

On Windows, create a Python environment, install `scripts/requirements.txt`, run `scripts/fetch_models.py --all`, and install the desktop dependencies with `npm ci` in `apps/desktop`. `scripts/fetch_twitchdownloader.py` fetches the pinned TwitchDownloader CLI. Packaging also requires the reviewed FFmpeg executable and its licensing files under `vendor/ffmpeg`, plus the pinned Microsoft runtime inputs prepared by `publish/prepare_microsoft_runtime.py`. The release build uses `scripts/package_installed.ps1` with a GitHub application-package URL.

## Licensing and source

Recall's code is licensed under [AGPL-3.0](LICENSE). Bundled tools and models retain their own licenses and notices.

The FFmpeg source mirror for Recall is [recall-ffmpeg-source](https://github.com/brady-balk3/recall-ffmpeg-source). Packaged FFmpeg binaries carry separate build-identity and license notices. The source mirror must be accessible to everyone receiving a public binary release.

## Contributions

Recall is maintained by one person and does not accept pull requests or outside code changes. Bug reports are welcome through Issues. You are free to fork the project under the terms of the AGPL-3.0 license.
