# DROID/OpenPI Lab Pipeline Site

This directory is a standalone docs site.

Public GitHub Pages URL after the workflow is enabled:

```text
https://wisc-hci.github.io/openpi/droid-openpi-pipeline/
```

## Jekyll path

Use this when Ruby development headers are available:

```bash
cd docs/droid-openpi-pipeline
bundle install
bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

Then open:

```text
http://127.0.0.1:4000/droid-openpi-pipeline/
```

## Local fallback path

Use this on machines where Ruby/Jekyll native gems cannot build:

```bash
cd docs/droid-openpi-pipeline
python3 preview.py --serve
```

Then open:

```text
http://127.0.0.1:4000/droid-openpi-pipeline/
```
