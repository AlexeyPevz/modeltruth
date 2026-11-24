#!/bin/sh
# Cross-builds release archives into dist/ with checksums.txt. Needs VERSION.
set -eu
: "${VERSION:?VERSION is required}"
mkdir -p dist
for target in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64; do
  os="${target%/*}"; arch="${target#*/}"
  name="modeltruth_${VERSION}_${os}_${arch}"
  bin="modeltruth"; [ "$os" = windows ] && bin="modeltruth.exe"
  mkdir -p "build/$name"
  CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" go build -trimpath \
    -ldflags "-s -w -X main.version=${VERSION}" \
    -o "build/$name/$bin" ./cmd/modeltruth
  cp LICENSE README.md "build/$name/"
  if [ "$os" = windows ]; then
    (cd build && zip -qr "../dist/$name.zip" "$name")
  else
    tar -C build -czf "dist/$name.tar.gz" "$name"
  fi
done
(cd dist && sha256sum * > checksums.txt)
