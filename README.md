# ContentArmour

A Python proof-of-concept for embedding and detecting a forensic watermark in grayscale frequency-domain image data.

## Run

```bash
python3 main.py
```

The script generates a synthetic screen image, adds a forensic watermark, simulates a phone-photo leak, runs true-source and wrong-room checks, and saves `content_armor_forensics.png`.

## Test Images

The repository includes:

- `test_image.jpeg`: RGB test image processed through grayscale watermarking.
- `latest_test_image.avif`: AVIF test image with a generated RGB armored output.
- `content_armor_latest_test_image.png`: four-panel forensic result dashboard.
