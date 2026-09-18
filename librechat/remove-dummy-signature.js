/**
 * Remove the DUMMY_SIGNATURE fallback from the google-genai code vendored
 * inside `@librechat/agents`.
 *
 * The constant is a REAL thought signature the langchain authors captured from
 * an unrelated API session (their get_weather / "Boston" test). When a replayed
 * gemini-3 tool call has no signature of its own -- which is always true for
 * the non-first calls of a parallel batch, since Gemini only signs the first --
 * the library substitutes this blob. Gemini then decodes it as genuine
 * reasoning state, and the model's thought summaries derail into that foreign
 * context ("the user asked for the weather in Boston"), looping when the
 * phantom goal cannot be satisfied. See langchain-ai/langchain-google#1570 for
 * the same constant being rejected outright on the Vertex path.
 *
 * A call that has no real signature must send none. Signatures that DO exist
 * are untouched -- capture, persistence and replay of the real ones is handled
 * by thought-signatures.patch.
 *
 * Exits non-zero if the expected code is not found, so a version bump that
 * moves the fallback fails the image build instead of silently shipping the
 * bug back in. Run after `npm ci`.
 */
const fs = require('fs');

const TARGETS = [
  'node_modules/@librechat/agents/dist/cjs/llm/google/utils/common.cjs',
  'node_modules/@librechat/agents/dist/esm/llm/google/utils/common.mjs',
];

/**
 * Target the return statement itself rather than the surrounding condition:
 * the condition is incidental formatting that a bundler can change, while
 * `return DUMMY_SIGNATURE;` is the behaviour being removed, and it appears
 * exactly once per file.
 */
const FALLBACK = /return DUMMY_SIGNATURE;/g;

let failed = false;
for (const target of TARGETS) {
  if (!fs.existsSync(target)) {
    console.error(`remove-dummy-signature: MISSING ${target}`);
    failed = true;
    continue;
  }
  const source = fs.readFileSync(target, 'utf8');
  const hits = source.match(FALLBACK);
  if (!hits) {
    console.error(`remove-dummy-signature: fallback NOT FOUND in ${target} — ` +
      'the vendored library changed; re-verify the fix before shipping.');
    failed = true;
    continue;
  }
  const patched = source.replace(FALLBACK, 'return "";');
  if (/return DUMMY_SIGNATURE;/.test(patched)) {
    console.error(`remove-dummy-signature: fallback STILL PRESENT after patching ${target}`);
    failed = true;
    continue;
  }
  fs.writeFileSync(target, patched);
  console.log(`remove-dummy-signature: patched ${target} (${hits.length} occurrence(s))`);
}
if (failed) process.exit(1);
