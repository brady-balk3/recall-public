// SPDX-License-Identifier: AGPL-3.0-or-later
// Recall shim for @huggingface/transformers. OpenCut's in-browser Whisper
// downloads model weights from the internet; Recall stays offline and already
// has word-level transcripts for every clip, so this path is disabled.
export type AutomaticSpeechRecognitionOutput = { text: string; chunks?: { text: string; timestamp: [number, number | null] }[] };
export type AutomaticSpeechRecognitionPipeline = (audio: Float32Array, options?: Record<string, unknown>) => Promise<AutomaticSpeechRecognitionOutput | AutomaticSpeechRecognitionOutput[]>;
export function pipeline(_task: string, _model: string, _options?: Record<string, unknown>): Promise<AutomaticSpeechRecognitionPipeline> {
  return Promise.reject(new Error("On-device transcription in the editor is off. Recall adds captions from its own transcript."));
}
export const env = { allowLocalModels: false, allowRemoteModels: false };
