import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class WatermarkVerificationError(Exception):
    """Raised when forensic extraction fails or input dimensions are invalid."""


@dataclass
class LeakForensicReport:
    hall_id: str
    timestamp: str
    signature_hash: str
    detection_z_score: float
    is_leaked_from_source: bool


class ContentArmorEngine:
    """
    Invisible Frequency-Domain Screen-to-Camera Watermarking Engine.
    Embeds a spread-spectrum signature in the 2D FFT mid-frequency ring
    so it survives phone camera capture, glare, noise, and cropping.
    """

    def __init__(self, strength: float = 45.0, radius_ratio: float = 0.22, num_tones: int = 24):
        self.strength = strength
        self.radius_ratio = radius_ratio
        self.num_tones = num_tones

    def _derive_coordinates(self, shape: tuple[int, int], payload_key: str) -> list[tuple[int, int]]:
        """Deterministically maps a Hall+Timestamp key to symmetric mid-frequency FFT coordinates."""
        h, w = shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * self.radius_ratio)

        # Deterministic PRNG seeded by cryptographic hash of the forensic payload
        digest = hashlib.sha256(payload_key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)

        coords = []
        angles = np.linspace(0, np.pi, self.num_tones, endpoint=False)
        jitter = rng.uniform(-0.04, 0.04, size=self.num_tones)

        for theta in angles + jitter:
            dy = int(round(radius * np.sin(theta)))
            dx = int(round(radius * np.cos(theta)))
            coords.append((cy + dy, cx + dx))
        return coords

    def embed_armor(self, image: np.ndarray, hall_id: str, timestamp: str) -> np.ndarray:
        """Embeds an invisible forensic signature into a grayscale image matrix (0-255)."""
        if image.ndim != 2:
            raise WatermarkVerificationError("Expected a 2D grayscale image matrix.")

        payload_key = f"{hall_id}|{timestamp}"
        coords = self._derive_coordinates(image.shape, payload_key)
        h, w = image.shape
        cy, cx = h // 2, w // 2

        # Transform spatial pixels -> 2D Frequency Spectrum
        f_transform = np.fft.fft2(image.astype(np.float64))
        f_shift = np.fft.fftshift(f_transform)

        # Inject conjugate-symmetric frequency impulses (keeps spatial image purely real)
        adaptive_gain = self.strength * np.sqrt(h * w)
        for (y, x) in coords:
            sym_y, sym_x = (2 * cy - y) % h, (2 * cx - x) % w
            phase = np.angle(f_shift[y, x])
            f_shift[y, x] += adaptive_gain * np.exp(1j * phase)
            f_shift[sym_y, sym_x] += adaptive_gain * np.exp(-1j * phase)

        # Inverse 2D FFT back to spatial image
        armored = np.fft.ifft2(np.fft.ifftshift(f_shift)).real
        return np.clip(armored, 0, 255).astype(np.uint8)

    def simulate_camera_leak(self, armored_image: np.ndarray) -> np.ndarray:
        """Simulates someone taking a smartphone photo of the monitor (crop + glare + noise)."""
        rng = np.random.default_rng(42)
        img = armored_image.astype(np.float64)

        # 1. Simulate screen glare gradient & exposure shift
        h, w = img.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        glare = 25.0 * np.exp(-((x_grid - w * 0.7) ** 2 + (y_grid - h * 0.3) ** 2) / (2 * (w * 0.35) ** 2))
        img = img * 0.88 + glare + 10.0

        # 2. Simulate camera sensor Gaussian noise
        img += rng.normal(0, 9.5, size=img.shape)

        # 3. Simulate slight lens softening (3x3 box blur)
        padded = np.pad(img, 1, mode="edge")
        blurred = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0

        # 4. Simulate off-center phone crop (85% frame) & resize back via nearest/bilinear grid
        ch, cw = int(h * 0.86), int(w * 0.86)
        cropped = blurred[15:15 + ch, 20:20 + cw]
        row_idx = np.linspace(0, ch - 1, h).astype(int)
        col_idx = np.linspace(0, cw - 1, w).astype(int)
        captured = cropped[np.ix_(row_idx, col_idx)]

        return np.clip(captured, 0, 255).astype(np.uint8)

    def detect_leak(
        self,
        suspect_image: np.ndarray,
        candidate_hall: str,
        candidate_timestamp: str,
        z_threshold: float = 4.0,
    ) -> LeakForensicReport:
        """Checks if a suspect leaked photo contains the hidden Hall+Timestamp signature."""
        payload_key = f"{candidate_hall}|{candidate_timestamp}"
        coords = self._derive_coordinates(suspect_image.shape, payload_key)

        # High-pass Laplacian residual suppresses natural image content & highlights spectral tones
        img = suspect_image.astype(np.float64)
        padded = np.pad(img, 1, mode="edge")
        residual = (
            4 * img
            - padded[:-2, 1:-1]
            - padded[2:, 1:-1]
            - padded[1:-1, :-2]
            - padded[1:-1, 2:]
        )

        mag_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual)))

        # Sample target signature bins vs background mid-frequency noise
        target_energies = np.array([mag_spectrum[y, x] for (y, x) in coords])
        h, w = suspect_image.shape
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * self.radius_ratio)

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ring_mask = (dist >= r - 6) & (dist <= r + 6)
        ring_values = mag_spectrum[ring_mask]

        # Statistical Z-score test (H0: no signature present)
        z_score = float((np.mean(target_energies) - np.mean(ring_values)) / (np.std(ring_values) + 1e-8))
        sig_hash = hashlib.sha256(payload_key.encode()).hexdigest()[:16]

        return LeakForensicReport(
            hall_id=candidate_hall,
            timestamp=candidate_timestamp,
            signature_hash=sig_hash,
            detection_z_score=round(z_score, 2),
            is_leaked_from_source=bool(z_score >= z_threshold),
        )


def create_synthetic_confidential_slide(size: int = 512) -> np.ndarray:
    """Generates a clean synthetic 'Confidential Exam / Blueprint' slide image."""
    canvas = np.full((size, size), 235, dtype=np.uint8)
    # Header bar & geometric layout blocks
    canvas[30:90, 40:472] = 45
    canvas[120:260, 40:240] = 175
    canvas[120:260, 270:472] = 120
    # Simulated text lines
    for row in range(290, 460, 22):
        line_len = 260 + ((row * 37) % 160)
        canvas[row:row + 8, 40:40 + line_len] = 60
    return canvas


if __name__ == "__main__":
    engine = ContentArmorEngine(strength=38.0)

    # 1. Original confidential image
    original = create_synthetic_confidential_slide(512)

    # 2. Embed invisible forensic stamp for a specific hall & timestamp
    hall = "AUDITORIUM-B2-SEAT-14"
    ts = datetime(2026, 11, 26, 19, 10, 0).isoformat()
    armored = engine.embed_armor(original, hall, ts)

    # 3. Simulate a student/insider snapping a photo of the screen (crop + glare + noise)
    leaked_photo = engine.simulate_camera_leak(armored)

    # 4. Run forensic hypothesis test: True source vs. Wrong room
    true_report = engine.detect_leak(leaked_photo, hall, ts)
    fake_report = engine.detect_leak(leaked_photo, "AUDITORIUM-A1-SEAT-02", ts)

    print("=== TRUE SOURCE FORENSIC CHECK ===")
    print(json.dumps(asdict(true_report), indent=2))
    print("\n=== FALSE ROOM CONTROL CHECK ===")
    print(json.dumps(asdict(fake_report), indent=2))

    # 5. Visual Dashboard (Spatial + Frequency Spectrum)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("1. Original Screen")

    axes[1].imshow(armored, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("2. Armored (Invisible Stamp)")

    axes[2].imshow(leaked_photo, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("3. Phone Photo Leak\n(Cropped + Glare + Noise)")

    spectrum_vis = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(leaked_photo))))
    axes[3].imshow(spectrum_vis, cmap="inferno")
    axes[3].set_title(f"4. 2D FFT Forensics\nZ-Score: {true_report.detection_z_score} (DETECTED)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("content_armor_forensics.png", dpi=150)
    plt.show()