#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HOME="$SCRIPT_DIR/.home"
export GEM_HOME="$SCRIPT_DIR/.gems"
export GEM_PATH="$SCRIPT_DIR/.gems"
export GEM_SPEC_CACHE="$SCRIPT_DIR/.gem-spec-cache"
export PATH="$SCRIPT_DIR/.gems/bin:$PATH"

mkdir -p "$HOME" "$GEM_HOME" "$GEM_SPEC_CACHE"

if ! command -v bundle >/dev/null 2>&1; then
  gem install --install-dir "$GEM_HOME" --bindir "$GEM_HOME/bin" bundler
fi

bundle config set path vendor/bundle
bundle install
bundle exec jekyll serve --host 127.0.0.1 --port 4000
