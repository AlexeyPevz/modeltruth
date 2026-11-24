#!/bin/sh
# Creates the GitHub release for CI_COMMIT_TAG with generated notes and uploads dist/*.
set -eu
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required}" "${CI_COMMIT_TAG:?}" "${CI_REPO:?}"
api="https://api.github.com/repos/$CI_REPO"
auth="Authorization: Bearer $GITHUB_TOKEN"
accept="Accept: application/vnd.github+json"
curl -fsS -H "$auth" -H "$accept" "$api/git/ref/tags/$CI_COMMIT_TAG" >/dev/null
body=$(jq -n --arg t "$CI_COMMIT_TAG" '{tag_name:$t,name:$t,generate_release_notes:true}')
id=$(curl -fsS -X POST -H "$auth" -H "$accept" "$api/releases" -d "$body" | jq -er .id)
for f in dist/*; do
  curl -fsS -X POST -H "$auth" -H "$accept" -H "Content-Type: application/octet-stream" \
    --data-binary "@$f" "https://uploads.github.com/repos/$CI_REPO/releases/$id/assets?name=$(basename "$f")" >/dev/null
  echo "uploaded $(basename "$f")"
done
