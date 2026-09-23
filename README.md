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

## Scan modes

| Mode | What it does | When to use it |
| --- | --- | --- |
| **Best quality** (default) | Transcribes the whole stream, reads facecam expressions, and checks on-screen text every 3 seconds. Also builds a complete searchable transcript. | Your normal choice. Slower, but it sees everything. |
| **Smart scan** | Skips the full transcript and facecam expressions, and checks on-screen text every 6 seconds. Only the clips it picks get transcribed for captions. | A quick first pass on a long VOD. It can miss moments that are mostly talking. |

> **Known issue in 0.1.0-beta.1:** Smart scan can return few or even zero clips on some videos, because without a transcript one of the judges rejects too much. Use **Best quality** until this is fixed in the next beta.

## Performance profiles

You choose how much of your PC Recall may use, in onboarding or later in Settings. Scans always run at below-normal priority and use Windows' efficiency mode, so the app you're using stays in front.

| Profile | CPU workers | RAM kept free | Use it when |
| --- | --- | --- | --- |
| **Keep PC responsive** | 1–4, leaving about 25% of cores free | 6 GB | You're gaming or streaming while it scans. |
| **Balanced** (recommended) | Physical cores minus 2 (2–12) | 4 GB | Everyday use. |
| **Full speed** | Physical cores minus 1 (up to 16) | 2 GB | You've walked away from the PC. |

### How the work is split

The heavy part of a scan, reading video frames for on-screen text, people and your facecam, is split across parallel **perception workers**. Recall sizes that pool from your hardware at the start of each scan:

1. **CPU pool.** One worker per *physical* core (hyperthreads don't help this workload), minus the profile's headroom. It's then capped so each worker gets about 1.5 GB of RAM while the profile's reserve stays free.
2. **GPU upgrade.** With an NVIDIA GPU, Recall measures free VRAM, keeps 2 GB back for Windows and your other apps, and allows about 1.2 GB per GPU worker. If at least two GPU workers fit, the pool moves to the GPU. Each GPU worker runs about 2–3× real time, against about 1× for a CPU worker. If fewer than two fit, it keeps the full CPU pool, because that's faster. The upgrade can never make a scan slower.
3. **One heavy model at a time.** Speech recognition and the clip judges run in later stages, one after another, never alongside the perception workers. Peak VRAM is therefore whichever of the two is larger, not the sum.

## AI models

Every model runs locally. The large ones are downloaded from Hugging Face at pinned, checksum-verified versions during setup; the small ones ship with the app.

| Model | What Recall uses it for | Size on disk | Approx. VRAM while running |
| --- | --- | --- | --- |
| [Qwen3-ASR 1.7B](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) (Q8) | Speech recognition for transcripts and captions | 2.5 GB | ~3 GB |
| [Qwen3 ForcedAligner 0.6B](https://huggingface.co/valoomba/Qwen3-ForcedAligner-0.6B-ONNX) | Word-level timing for captions | 3.7 GB | ~4 GB |
| [Qwen3-4B Instruct 2507](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF) (Q4_K_M) | Text judge: does this moment make sense and land on its own? | 2.5 GB | ~3 GB |
| [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) (Q4_K_M, with vision) | Visual judge: looks at frames from candidate clips | 3.4 GB | ~3–4 GB |
| YOLOX-s / YOLOX-tiny | Person detection for facecam framing | small | part of perception workers |
| PaddleOCR PP-OCRv6 (small) | Reading on-screen text such as kill feeds and win banners | small | part of perception workers |
| MediaPipe Face Landmarker | Facecam expressions (Best quality only) | small | CPU |
| YAMNet | Audio events: laughter, screams, combat sounds and intense music | small | CPU |
| Silero VAD | Finding speech so transcription skips silence | 2 MB | CPU |
| WeSpeaker ResNet34 | Telling speakers apart | 27 MB | CPU |
| MiniLM | Meaning-based search over your stream transcripts | small | CPU |
| Recall scoring models | Ranking candidate moments and a second-look filter | 3.5 MB | CPU |

The large models total about 12 GB. VRAM figures are estimates from model size. On the test machine, perception workers measured about 1 GB of VRAM each.

## CPU-only mode

Recall runs on machines without a supported GPU, and it tells you when it's in CPU mode. Every stage, including speech recognition and the judges, falls back to the CPU. AMD and Intel GPUs aren't accelerated yet, so they use CPU mode too.

CPU-only mode is **very limited**. A full Best quality scan of a multi-hour stream can take many times longer than on an NVIDIA GPU, and the judges are especially slow. Keep VODs short and expect to leave it running. (Smart scan would be faster, but see the known issue above.) This path has not been tested on a machine without an NVIDIA GPU yet.

## Install

Download the Windows installer from this repository's Releases page when a release is available. The small setup program downloads a checksum-verified application package from GitHub and the pinned large model weights from Hugging Face. You do not need to download model files manually. The first install needs an internet connection and enough disk space for the application, models, and your source videos.

**Expect two security warnings.** Recall is new and its installer isn't code-signed yet, so Windows and your browser haven't seen it often enough to trust it automatically:

- **Chrome or Edge** may say the file *"isn't commonly downloaded"*. In Chrome, open the downloads list and choose **Keep** (under the ⋮ menu if needed). In Edge, hover the download, click **⋯ → Keep**, then **Show more → Keep anyway**.
- **Windows** may then say *"Windows protected your PC"*. Click **More info → Run anyway**.

The installer checks the SHA-256 of everything it downloads before installing, and the source for every file is in this repository.

Your recordings, scan data, and exported clips are stored locally. Downloading a Twitch VOD, application package, or model uses the network.

## Build from source

The application source lives in `apps/`, `core/`, `engines/`, and `pipeline/`. The desktop uses Electron and React; the local engine uses Python. Model weights, downloaded runtimes, recordings, and generated builds are excluded from Git.

On Windows, create a Python environment, install `scripts/requirements.txt`, run `scripts/fetch_models.py --all`, and install the desktop dependencies with `npm ci` in `apps/desktop`. `scripts/fetch_twitchdownloader.py` fetches the pinned TwitchDownloader CLI. Packaging also requires the reviewed FFmpeg executable and its licensing files under `vendor/ffmpeg`, plus the pinned Microsoft runtime inputs prepared by `publish/prepare_microsoft_runtime.py`. The release build uses `scripts/package_installed.ps1` with a GitHub application-package URL.

## Licensing and source

Recall's code is licensed under [AGPL-3.0](LICENSE). Bundled tools and models retain their own licenses and notices.

The FFmpeg source mirror for Recall is [recall-ffmpeg-source](https://github.com/brady-balk3/recall-ffmpeg-source). Packaged FFmpeg binaries carry separate build-identity and license notices. The source mirror must be accessible to everyone receiving a public binary release.

## Contributions

Recall is maintained by one person and does not accept pull requests or outside code changes. Bug reports are welcome through Issues. You are free to fork the project under the terms of the AGPL-3.0 license.
