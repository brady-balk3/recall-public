# Recall

**Turn a five-hour stream into clips you can post — without scrubbing through it yourself.**

Recall is a free Windows app for streamers and the people who clip for them. Point it at a Twitch VOD or a recording on your PC, and it watches the whole thing for you: the big reactions, the funny exchanges, the clutch plays, the moments chat went off. It hands back a deck of short vertical clips, already framed for TikTok, YouTube Shorts and Reels, with your facecam placed and captions timed to what you said.

You stay in charge. Every clip opens in a review editor where you keep or pass, nudge the start and end, fix the framing, or correct a caption before anything is exported.

## What it does

- **Finds moments, not just loud bits.** It combines what's on screen, what you say, how you react, and what chat does, then picks the moments worth watching on their own.
- **Frames for vertical.** It detects your facecam and stacks it with the gameplay for 9:16 video.
- **Captions automatically.** Word-timed captions from on-device speech recognition, and you can edit them.
- **Keeps you in control.** Review, trim, reframe and re-render any clip before it leaves the app.
- **Runs entirely on your PC.** No account, no upload, no subscription. Your recordings, scans and clips stay on your machine; the network is used only to download a Twitch VOD you ask for and to fetch the app and its models during setup.

## System requirements

| | Recommended | Notes |
| --- | --- | --- |
| **OS** | Windows 11 or Windows 10, 64-bit | No macOS or Linux build. |
| **GPU** | NVIDIA GeForce RTX with 8 GB+ VRAM | Speech recognition, detection and the clip judges run on the GPU through CUDA 12. Keep your NVIDIA driver current. |
| **RAM** | 32 GB (16 GB minimum) | |
| **Disk** | 25 GB free for the install, plus room for your videos | The app takes about 7 GB and the AI models about 12 GB. A downloaded Twitch VOD needs a few GB per hour of stream. |
| **Internet** | Needed for setup and for Twitch VODs | Once installed, scanning and exporting local recordings is designed to work offline. |

**Without an NVIDIA GPU** Recall falls back to the CPU. That path hasn't been tested on a machine without an NVIDIA card, and scans will be much slower.

**Tested on:** Windows 11 Pro, AMD Ryzen 7 7800X3D, 32 GB RAM, NVIDIA RTX 4070 Ti SUPER (16 GB). Other setups should work within the table above, but haven't been verified yet. If yours doesn't, please open an issue.

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
