# Use script to download

Use the following command to download model:
```python
python download_model.py
```

Or use auto downloader to download, which can automatically resume downloading:
```bash
bash auto_download.sh
```

# Use hfd.sh to download

It is more recommended to use `hfd.sh` to download, which is much faster and more stable:

```bash
bash hfd.sh ByteDance-Seed/BAGEL-7B-MoT --local-dir your/local/dir -x 8 -j 8
```