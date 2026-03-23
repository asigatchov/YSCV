#!/usr/bin/env bash
set -euo pipefail

# Bump version across all workspace crates.
# Usage: ./scripts/bump-version.sh <new_version>

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>" >&2
    exit 1
fi

VERSION="$1"

# Validate semver format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid version format: $VERSION (expected X.Y.Z)" >&2
    exit 1
fi

echo "Bumping all crates to version $VERSION"

# Update workspace version
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
rm -f Cargo.toml.bak

echo "Updated workspace Cargo.toml"
echo "Run 'cargo check --workspace' to verify."
