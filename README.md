# ContentArmour

A Python proof-of-concept for embedding and detecting a forensic watermark in grayscale frequency-domain image data.

<img width="2385" height="485" alt="content_armor_test_image" src="https://github.com/user-attachments/assets/839d01b6-4d2c-4ebc-98c0-aa477d6bf10a" />

<img width="2385" height="415" alt="content_armor_latest_test_image" src="https://github.com/user-attachments/assets/da49e2f0-21dc-43c2-b257-2f8c39fb14e4" />


```bash
python3 main.py
```

The script generates a synthetic screen image, adds a forensic watermark, simulates a phone-photo leak, runs true-source and wrong-room checks, and saves `content_armor_forensics.png`.

## Test Images
The repository includes:

- `test_image.jpeg`: RGB test image processed through grayscale watermarking.
- `latest_test_image.avif`: AVIF test image with a generated RGB armored output.
- `content_armor_latest_test_image.png`: four-panel forensic result dashboard.
