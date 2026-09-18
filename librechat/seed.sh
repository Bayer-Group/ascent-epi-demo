#!/bin/sh
# Seed LibreChat: a demo account, and an agent with the ASCENT MCP tools
# already attached, so the UI is usable the moment it comes up.
#
# Two steps with different needs. Creating the user talks to Mongo directly
# through LibreChat's own CLI; creating the agent goes through its HTTP API and
# so has to wait for the service. The second half lives in seed.js because this
# image ships no curl.
#
# Never fails the stack: a missing agent is a degraded demo, not a broken one.

echo "seed: creating user ${CHAT_USER:-demo@ascent.local}"
# --email-verified is required. Without it create-user prompts on stdin, finds
# no TTY, and hangs forever.
npm run create-user -- \
  "${CHAT_USER:-demo@ascent.local}" Demo demo "${CHAT_PASSWORD:-ascentdemo}" \
  --email-verified=true || true

node /seed/seed.js || true
exit 0
