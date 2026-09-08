"""
demo.py  —  ALM Inference Demo
================================
Run inference on any audio file using:
  • AudioEncoder         → single-label classification + embedding extraction
  • MultiLabelAudioEncoder → multi-label event detection (mixed sounds)

Usage
-----
  # Single-label inference (default)
  python demo.py --audio path/to/sound.wav

  # Multi-label inference
  python demo.py --audio path/to/sound.wav --mode multilabel

  # Mix two audio files and predict (multilabel only)
  python demo.py --audio path/to/sound1.wav --mix path/to/sound2.wav --mix-alpha 0.5 --mode multilabel

  # Add Gaussian noise (SNR in dB) then predict
  python demo.py --audio path/to/sound.wav --noise-snr 10 --mode multilabel

  # Use a specific checkpoint
  python demo.py --audio path/to/sound.wav --ckpt models/alm_audio_encoder_best.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.audio_encoder import AudioEncoder, MultiLabelAudioEncoder

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Default checkpoint paths ───────────────────────────────────────────────────
DEFAULT_SINGLELABEL_CKPT  = ROOT / "models" / "alm_audio_encoder_best.pth"
DEFAULT_MULTILABEL_CKPT   = ROOT / "models" / "alm_multilabel_best.pth"


DEMO_AUDIO_DIR = (
    ROOT / "outputs" / "demo_audio"
)

# ── Single-label: 58-class vocabulary (sorted, from metadata.csv) ──────────────
SINGLE_LABEL_CLASSES = [
    "air_conditioner", "airplane", "breathing", "brushing_teeth", "can_opening",
    "car_horn", "cat", "chainsaw", "children_playing", "chirping_birds",
    "church_bells", "clapping", "clock_alarm", "clock_tick", "coughing",
    "cow", "crackling_fire", "crickets", "crow", "crying_baby",
    "dog", "dog_bark", "door_wood_creaks", "door_wood_knock", "drilling",
    "drinking_sipping", "engine", "engine_idling", "fireworks", "footsteps",
    "frog", "glass_breaking", "gun_shot", "hand_saw", "helicopter",
    "hen", "insects", "jackhammer", "keyboard_typing", "laughing",
    "mouse_click", "pig", "pouring_water", "rain", "rooster",
    "sea_waves", "sheep", "siren", "sneezing", "snoring",
    "street_music", "thunderstorm", "toilet_flush", "train", "vacuum_cleaner",
    "washing_machine", "water_drops", "wind",
]

# ── Multi-label thresholds ─────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.35
CLASS_THRESHOLDS  = {
    "gun_shot"  : 0.55,
    "siren"     : 0.50,
    "fireworks" : 0.45,
    "train"     : 0.50,
}

# ── Audio preprocessing constants ─────────────────────────────────────────────
TARGET_SR        = 24_000       # 24 kHz  (matches preprocessing pipeline)
TARGET_SAMPLES   = 120_000     # 5 s × 24 kHz
SINGLELABEL_SR   = 16_000      # AudioTextDataset uses 16 kHz for MEL


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_standardize(audio_path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Load any audio file and standardize it following the ALM preprocessing pipeline:
      1. Load waveform
      2. Convert stereo → mono
      3. Resample to target_sr
      4. Trim / zero-pad to exactly 5 seconds
      5. Normalize amplitude to [-1, 1]

    Returns:
        wav  (1, target_samples)  float32 tensor
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav, sr = torchaudio.load(str(audio_path))

    # Step 2 — mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Step 3 — resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Step 4 — trim / pad to 5 s
    n = target_sr * 5
    if wav.shape[1] < n:
        wav = F.pad(wav, (0, n - wav.shape[1]))
    else:
        wav = wav[:, :n]

    # Step 5 — amplitude normalization
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak

    return wav   # (1, target_samples)


def wav_to_mel_singlelabel(wav: torch.Tensor) -> torch.Tensor:
    """
    Convert waveform to log-mel spectrogram for the single-label AudioEncoder.
    Matches AudioTextDataset._to_melspec() in script/audio_dataset.py.

    Config: sr=16k, n_fft=1024, hop=512, n_mels=128, f_max=8k
    Norm  : min-max → [-1, 1]

    Args:
        wav  (1, samples) at 16 kHz

    Returns:
        mel  (1, 128, T)
    """
    mel_tf     = T.MelSpectrogram(
        sample_rate = SINGLELABEL_SR,
        n_fft       = 1024,
        hop_length  = 512,
        n_mels      = 128,
        f_min       = 0,
        f_max       = 8_000,
    )
    amp_to_db  = T.AmplitudeToDB(stype='power', top_db=80)

    mel     = mel_tf(wav)
    log_mel = amp_to_db(mel)

    # Min-max normalize to [-1, 1]
    lo, hi  = log_mel.min(), log_mel.max()
    log_mel = (log_mel - lo) / (hi - lo + 1e-8)   # [0, 1]
    log_mel = log_mel * 2 - 1                       # [-1, 1]

    return log_mel   # (1, 128, T)


def wav_to_mel_multilabel(wav: torch.Tensor) -> torch.Tensor:
    """
    Convert waveform to log-mel spectrogram for the MultiLabelAudioEncoder.
    Matches MultiLabelAudioDataset.__getitem__() in script/audio_dataset.py.

    Config: sr=24k, n_fft=1024, hop=256, n_mels=128, f_min=50, f_max=12k
    Norm  : log(mel + eps) → z-score

    Args:
        wav  (1, samples) at 24 kHz

    Returns:
        mel  (1, 1, 128, T)  — extra batch-channel dim for direct model input
    """
    mel_tf = T.MelSpectrogram(
        sample_rate = TARGET_SR,
        n_fft       = 1024,
        hop_length  = 256,
        n_mels      = 128,
        f_min       = 50,
        f_max       = 12_000,
    )

    mel = mel_tf(wav)
    mel = (mel + 1e-9).log()
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return mel.unsqueeze(0)   # (1, 1, 128, T)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_singlelabel_encoder(ckpt_path: Path, device: str) -> tuple[AudioEncoder, list[str]]:
    """
    Load a saved AudioEncoder checkpoint.

    Returns:
        model         – AudioEncoder in eval mode
        label_classes – list[str] ordered by class index
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    label_classes = ckpt.get("label_classes", SINGLE_LABEL_CLASSES)
    num_classes   = len(label_classes)

    cfg   = ckpt.get("config", {})
    model = AudioEncoder(
        embed_dim   = cfg.get("embed_dim", 512),
        num_classes = num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  ✅ Single-label encoder loaded  |  {num_classes} classes  |  {ckpt_path.name}")
    return model, label_classes


def load_multilabel_encoder(ckpt_path: Path, device: str) -> tuple[MultiLabelAudioEncoder, list[str]]:
    """
    Load a saved MultiLabelAudioEncoder checkpoint.

    Returns:
        model         – MultiLabelAudioEncoder in eval mode
        label_classes – list[str] ordered by class index
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    model = MultiLabelAudioEncoder(
        embed_dim   = cfg["embed_dim"],
        num_heads   = cfg.get("num_heads", 8),
        num_layers  = cfg.get("num_layers", 4),
        ff_dim      = cfg.get("ff_dim", 1024),
        num_classes = cfg["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_classes = ckpt.get("label_classes", [])
    print(f"  ✅ Multi-label encoder loaded   |  {cfg['num_classes']} classes  |  {ckpt_path.name}")
    return model, label_classes


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE — SINGLE-LABEL
# ══════════════════════════════════════════════════════════════════════════════

def singlelabel_inference(
    model         : AudioEncoder,
    wav           : torch.Tensor,
    label_classes : list[str],
    device        : str,
    top_k         : int = 5,
) -> dict:
    """
    Run single-label classification + embedding extraction.

    Returns:
        {
          'pred_label'  : str,
          'confidence'  : float,
          'top_k'       : [(label, prob), ...],
          'embedding'   : np.ndarray  (embed_dim,)  L2-normalised
        }
    """
    # Single-label encoder uses 16 kHz for MEL
    wav_16k = torchaudio.functional.resample(wav, TARGET_SR, SINGLELABEL_SR)
    mel     = wav_to_mel_singlelabel(wav_16k).unsqueeze(0).to(device)   # (1, 1, 128, T)

    with torch.no_grad():
        logits = model(mel)                                  # (1, num_classes)
        probs  = F.softmax(logits, dim=-1).squeeze(0)       # (num_classes,)

    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item()
    pred_label  = label_classes[pred_idx] if label_classes else str(pred_idx)

    top_k_results = sorted(
        zip(label_classes, probs.cpu().tolist()),
        key=lambda x: -x[1]
    )[:top_k]

    # ── Embedding extraction (bypass classifier) ──────────────────────────────
    with torch.no_grad():
        x = model.backbone(mel)
        x = model.pool(x)
        x = x.flatten(1)
        x = model.projector(x)
        embedding = F.normalize(x, dim=-1).squeeze(0).cpu().numpy()

    return {
        "pred_label" : pred_label,
        "confidence" : confidence,
        "top_k"      : top_k_results,
        "embedding"  : embedding,
        "all_probs"  : dict(zip(label_classes, probs.cpu().tolist())),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE — MULTI-LABEL
# ══════════════════════════════════════════════════════════════════════════════

def multilabel_inference(
    model         : MultiLabelAudioEncoder,
    wav           : torch.Tensor,
    label_classes : list[str],
    device        : str,
    top_k         : int = 5,
) -> dict:
    """
    Run multi-label event detection with hybrid threshold / top-1 fallback.

    Returns:
        {
          'detected_events' : ['gun_shot', 'dog'],
          'probabilities'   : {'gun_shot': 0.82, ...},
          'decision_mode'   : 'threshold' | 'top_1_fallback',
          'top_k'           : [(label, prob), ...],
          'embedding'       : np.ndarray  (embed_dim,)  L2-normalised
        }
    """
    id2label = {i: lbl for i, lbl in enumerate(label_classes)}
    mel      = wav_to_mel_multilabel(wav).to(device)   # (1, 1, 128, T)

    with torch.no_grad():
        logits    = model(mel)                                        # (1, num_classes)
        probs_t   = torch.sigmoid(logits).squeeze(0).cpu()           # (num_classes,)
        embedding = model.encode(mel).squeeze(0).cpu().numpy()       # (embed_dim,)

    probs = probs_t.numpy()

    # ── Hybrid threshold detection ────────────────────────────────────────────
    active_ids = [
        i for i, p in enumerate(probs)
        if p > CLASS_THRESHOLDS.get(id2label[i], DEFAULT_THRESHOLD)
    ]
    if not active_ids:
        active_ids    = [int(np.argmax(probs))]
        decision_mode = "top_1_fallback"
    else:
        decision_mode = "threshold"

    top_k_results = sorted(
        zip(label_classes, probs.tolist()),
        key=lambda x: -x[1]
    )[:top_k]

    return {
        "detected_events" : [id2label[i] for i in active_ids],
        "probabilities"   : {id2label[i]: float(probs[i]) for i in range(len(id2label))},
        "decision_mode"   : decision_mode,
        "top_k"           : top_k_results,
        "embedding"       : embedding,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO MANIPULATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mix_audio(wav1: torch.Tensor, wav2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Mix two waveforms: mixed = wav1 + alpha * wav2, then peak-normalize.
    Both wavs must be (1, N_SAMPLES) at the same sample rate.
    """
    mixed = wav1 + alpha * wav2
    peak  = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def add_background_noise(wav: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """
    Add white Gaussian noise at a given Signal-to-Noise Ratio (in dB).
    """
    signal_power = wav.pow(2).mean()
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = torch.randn_like(wav) * noise_power.sqrt()
    return (wav + noise).clamp(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_banner(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def print_singlelabel_result(result: dict, audio_path: str):
    print_banner("SINGLE-LABEL CLASSIFICATION RESULT")
    print(f"  File       : {Path(audio_path).name}")
    print(f"  Prediction : {result['pred_label'].upper()}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"\n  Top-{len(result['top_k'])} Predictions:")
    for rank, (label, prob) in enumerate(result['top_k'], 1):
        bar = "█" * int(prob * 30)
        print(f"    {rank}. {label:<22}  {bar:<30}  {prob:.2%}")
    print(f"\n  Embedding  : shape={result['embedding'].shape}  "
          f"norm={np.linalg.norm(result['embedding']):.4f}")


def print_multilabel_result(result: dict, audio_path: str, scenario: str = ""):
    title = f"MULTI-LABEL EVENT DETECTION  {('— ' + scenario) if scenario else ''}"
    print_banner(title)
    print(f"  File           : {Path(audio_path).name}")
    print(f"  Detected Events: {result['detected_events']}")
    print(f"  Decision Mode  : {result['decision_mode']}")
    print(f"\n  Top-{len(result['top_k'])} Probabilities:")
    for label, prob in result['top_k']:
        thresh = CLASS_THRESHOLDS.get(label, DEFAULT_THRESHOLD)
        marker = "✓" if prob > thresh else "✗"
        bar    = "█" * int(prob * 30)
        print(f"    {marker} {label:<22}  {bar:<30}  {prob:.2%}  (thr={thresh:.2f})")
    print(f"\n  Embedding      : shape={result['embedding'].shape}  "
          f"norm={np.linalg.norm(result['embedding']):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_singlelabel(args, device: str):
    """Full single-label inference pipeline."""
    ckpt_path = Path(args.ckpt) if args.ckpt else DEFAULT_SINGLELABEL_CKPT

    print_banner("ALM — Single-Label Audio Encoder")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {ckpt_path.name}")

    model, label_classes = load_singlelabel_encoder(ckpt_path, device)

    # ── Load & preprocess audio ───────────────────────────────────────────────
    print(f"\n  Loading audio: {args.audio}")
    wav = load_and_standardize(args.audio, target_sr=TARGET_SR)

    # Optional: add noise before inference
    if args.noise_snr is not None:
        print(f"  Adding noise at SNR = {args.noise_snr} dB")
        wav = add_background_noise(wav, snr_db=args.noise_snr)
        noise_path = make_noise_filename(args.audio , args.noise_snr)

        save_waveform(wav , noise_path , sample_rate=TARGET_SR)

    # Optional: mix with a second file
    if args.mix:
        print(f"  Mixing with: {args.mix}  (alpha={args.mix_alpha})")
        wav2 = load_and_standardize(args.mix, target_sr=TARGET_SR)
        wav  = mix_audio(wav, wav2, alpha=args.mix_alpha)
        mix_path = make_mix_filename(args.audio , args.mix , args.mix_alpha)
        save_waveform(wav , mix_path , sample_rate=TARGET_SR)
    # ── Run inference ─────────────────────────────────────────────────────────
    result = singlelabel_inference(model, wav, label_classes, device, top_k=args.top_k)
    print_singlelabel_result(result, args.audio)


def run_multilabel(args, device: str):
    """Full multi-label inference pipeline with 3 demo scenarios."""
    ckpt_path = Path(args.ckpt) if args.ckpt else DEFAULT_MULTILABEL_CKPT

    print_banner("ALM — Multi-Label Audio Encoder")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {ckpt_path.name}")

    model, label_classes = load_multilabel_encoder(ckpt_path, device)

    # ── Load & preprocess PRIMARY audio ───────────────────────────────────────
    print(f"\n  Loading audio: {args.audio}")
    wav = load_and_standardize(args.audio, target_sr=TARGET_SR)

    # ── SCENARIO 1: Clean single-sound ────────────────────────────────────────
    result = multilabel_inference(model, wav, label_classes, device, top_k=args.top_k)
    print_multilabel_result(result, args.audio, "Scenario 1: Clean Audio")

    # ── SCENARIO 2: Mixed audio (if --mix provided) ───────────────────────────
    if args.mix:
        print(f"\n  Mixing with: {args.mix}  (alpha={args.mix_alpha})")
        wav2   = load_and_standardize(args.mix, target_sr=TARGET_SR)
        mixed  = mix_audio(wav, wav2, alpha=args.mix_alpha)
        mix_path = make_mix_filename(
            args.audio,
            args.mix,
            args.mix_alpha
        )
        print("saving mixed Audio ..")
        save_waveform(
            mixed,
            mix_path,
            sample_rate=TARGET_SR
        )
        result = multilabel_inference(model, mixed, label_classes, device, top_k=args.top_k)
        print_multilabel_result(result, args.audio, f"Scenario 2: Mixed Audio (alpha={args.mix_alpha})")

    # ── SCENARIO 3: Noisy audio (if --noise-snr provided) ────────────────────
    if args.noise_snr is not None:
        print(f"\n  Adding noise at SNR = {args.noise_snr} dB")
        noisy_wav = add_background_noise(wav, snr_db=args.noise_snr)
        noise_path = make_noise_filename(
            args.audio,
            args.noise_snr
        )
        save_waveform(
            noisy_wav,
            noise_path,
            sample_rate=TARGET_SR
        )
        
        result    = multilabel_inference(model, noisy_wav, label_classes, device, top_k=args.top_k)
        print_multilabel_result(result, args.audio, f"Scenario 3: Noisy Audio (SNR={args.noise_snr}dB)")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALM Inference Demo — single-label & multi-label audio classification",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--audio", required=True, type=str,
        help="Path to the primary input audio file (.wav, .mp3, .flac, etc.)",
    )
    parser.add_argument(
        "--mode", choices=["singlelabel", "multilabel"], default="singlelabel",
        help="Inference mode:\n"
             "  singlelabel  — single-class classification + embedding (default)\n"
             "  multilabel   — multi-event detection for mixed sounds",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to a specific checkpoint (.pth). "
             "Defaults to models/alm_audio_encoder_best.pth or models/alm_multilabel_best.pth",
    )
    parser.add_argument(
        "--mix", type=str, default=None,
        help="(multilabel) Path to a second audio file to mix with --audio",
    )
    parser.add_argument(
        "--mix-alpha", type=float, default=0.5,
        help="(multilabel) Volume ratio for the mixed file: mixed = audio + alpha*mix  (default: 0.5)",
    )
    parser.add_argument(
        "--noise-snr", type=float, default=None,
        help="Add Gaussian noise at this SNR in dB before inference (e.g., 10.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top predictions to display (default: 5)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on: 'cuda' or 'cpu'. Auto-detected if not set.",
    )
    return parser.parse_args()

def save_waveform(wav: torch.Tensor, output_path: str, sample_rate: int = 24000):
    """
    Save a waveform tensor as a playable WAV file.

    Args:
        wav:
            Tensor shaped (1, N) or (N,)
        output_path:
            Destination .wav path
        sample_rate:
            Sampling rate used by the ALM pipeline
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Move to CPU and convert to NumPy
    wav_np = wav.detach().cpu().squeeze().numpy()

    # Keep waveform in valid WAV amplitude range
    wav_np = np.clip(wav_np, -1.0, 1.0)

    # Write WAV
    sf.write(
        str(output_path),
        wav_np,
        sample_rate,
        subtype="PCM_16"
    )

    print(f"   💾 Audio saved → {output_path}")


def make_mix_filename(audio1 : str | Path , audio2 : str | Path , alpha : float) -> Path:
    name1 = Path(audio1).stem
    name2 = Path(audio2).stem

    return (DEMO_AUDIO_DIR /f"{name1}_PLUS_{name2}_alpha_{alpha}.wav")

def make_noise_filename(audio : str | Path , snr_db : float)-> Path:
    name = Path(audio).stem

    return (DEMO_AUDIO_DIR/f"{name}_NOISE_SNR_{snr_db}dB.wav")

def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  ALM — Audio Language Model  |  Inference Demo")
    print(f"{'='*60}")

    if args.mode == "singlelabel":
        run_singlelabel(args, device)
    else:
        run_multilabel(args, device)

    print("\n" + "═" * 60)
    print("  Done.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
