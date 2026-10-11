#!/bin/bash
# Advance every active Tuning Engine run: upload the next checkpoint's data, start its Forge job,
# finish runs, and clean up after Stop. A no-op when nothing is training.
APP_ID="$1"
curl -fsS -m 25 -X POST -H "Authorization: Bearer $APP_TOKEN" -H "Content-Type: application/json" \
  --data '{}' "${API_BASE_URL:-http://localhost:8080}/api/apps/$APP_ID/service/internal/tick" >/dev/null
