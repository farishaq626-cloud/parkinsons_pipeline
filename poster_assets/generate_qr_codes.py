"""Generate high-resolution QR codes for poster research metadata."""

from __future__ import annotations

from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "figures"
QR_TARGETS = {
    "github_qr.png": "https://github.com/farishaq626-cloud/parkinsons_pipeline",
    "zenodo_qr.png": "https://doi.org/10.5281/zenodo.21225810",
}


def generate_qr_code(url: str, destination: Path) -> Path:
    """Create a high-resolution, error-corrected QR code.

    Args:
        url: Web address encoded by the QR code.
        destination: PNG output path.

    Returns:
        Path to the generated PNG file.
    """
    qr_code = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=24,
        border=4,
    )
    qr_code.add_data(url)
    qr_code.make(fit=True)
    image = qr_code.make_image(fill_color="black", back_color="white").get_image()
    image.save(destination, format="PNG", dpi=(600, 600))
    return destination


def main() -> None:
    """Generate the GitHub and Zenodo QR-code assets."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename, url in QR_TARGETS.items():
        output_path = generate_qr_code(url, OUTPUT_DIRECTORY / filename)
        print(f"Generated {output_path} -> {url}")


if __name__ == "__main__":
    main()
