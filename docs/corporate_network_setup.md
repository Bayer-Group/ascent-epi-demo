# Running this on a corporate network (TLS-intercepting proxy)

A quick runbook, not an explanation. For the full rationale, read the comments
in [`docker-compose.tls-proxy.yml`](../docker-compose.tls-proxy.yml) — this
page just tells you what to type.

## Am I affected?

Only if you see one of these on an otherwise-normal `docker compose up`:

| Symptom | Where |
|---|---|
| `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` | `coder-init` logs |
| `ConnectionError: Network error: Request middleware error: error sending request` | `coder-init` logs |
| Stack starts fine, then every chat message fails with `UNABLE_TO_VERIFY_LEAF_SIGNATURE` | LibreChat, in the browser |

If you don't see any of these, **stop — you don't need this page**, and adding
the overlay for no reason just adds a rebuild.

## Fix it — one time per laptop

**1. Build a CA bundle: public roots + your company's proxy CA.**

```bash
mkdir -p certs
docker run --rm python:3.12-slim cat /etc/ssl/certs/ca-certificates.crt > certs/ca-bundle.crt
```

Start from a real image's bundle, not a bundle of just your company's CA —
drop the public roots and every *other* HTTPS call in the container breaks
instead.

**2. Find out which CA(s) you actually need.**

Interception is usually per-host, so don't assume — check the specific host
that's failing (`coder-init`'s error names it; for the HuggingFace case it's
usually the CDN, not the API host):

```bash
openssl s_client \
  -connect "${TLS_HOST:?Set TLS_HOST to the hostname in the failing request}:443" \
  -servername "$TLS_HOST" </dev/null 2>/dev/null | openssl x509 -noout -issuer
```

- Issuer is a public CA (DigiCert, Amazon, Let's Encrypt, ...) → this isn't
  your problem, look elsewhere.
- Issuer names your employer → that's the CA. Note its common name.

**3. Append every CA in the chain to the bundle** — not just the leaf
certificate. A proxy that only hands you its leaf still needs its intermediate
and root present locally, or no chain can be built.

macOS, once you have the common name:

```bash
security find-certificate -a -c "${PROXY_CA_NAME:?Set PROXY_CA_NAME to your trusted proxy CA common name}" -p \
  /Library/Keychains/System.keychain >> certs/ca-bundle.crt
```

Linux, wherever your distro or IT put it:

```bash
cat "${PROXY_CA_FILE:?Set PROXY_CA_FILE to your IT-provided PEM certificate path}" >> certs/ca-bundle.crt
```

Repeat for each certificate in the chain your proxy presents.

**4. Set or update `COMPOSE_FILE` in `.env`** (do not add duplicate entries):

```dotenv
COMPOSE_FILE=docker-compose.yml:docker-compose.tls-proxy.yml
```

From here on, plain `docker compose up` picks it up automatically — no need
to type `-f` flags every time.

**5. Bring the stack up.**

```bash
docker compose up
```

The overlay selects a CA-aware Dockerfile for `librechat`, so that image needs
one rebuild when switching the overlay on or off. Backend and medical-coder
are also built locally, but this overlay adds runtime trust mounts for them;
it does not add certificate handling to their Python dependency build layers.
If those builds fail certificate verification, configure build-time trust with
your organization's approved mechanism. Do not disable TLS verification.

## Verify it worked

```bash
docker compose logs coder-init medical-coder backend | grep -iE "certificate_verify|sslerror|middleware error"
```

No matching output only means these errors were not found in those logs. Check
service status and probes, then run the [functional smoke task](../README.md#try-a-first-task):

```bash
curl localhost:8000/api/public/health     # "ok"
curl localhost:8000/mcp/health            # MCP registration/mounting and Redis only
```

## If it still fails

- Re-run step 2 against *every* host that's failing, not just the first one.
  A proxy can intercept the model CDN but not the API host, so testing the
  wrong host tells you nothing.
- Confirm you appended the whole chain (step 3), not just one certificate.
- If `docker compose build` itself fails (not just runtime) with a certificate
  error — e.g. `apk` or `npm ci` failing inside the LibreChat build — inspect
  the CA-aware `librechat/Dockerfile.tls-proxy` build and verify that
  `certs/ca-bundle.crt` exists, is non-empty, and contains the required chain.
  Python image build failures need their own build-time trust configuration.

## Notes

- `certs/` is gitignored. **Never commit your company's CA** — which CA you
  need is a property of the network you're on, and committing one would ask
  every other user of this repo to trust a certificate for a network they
  aren't on.
- This whole page is opt-in and inert for anyone not on an intercepting
  network: if `docker compose up` already works for you, none of this loads.
