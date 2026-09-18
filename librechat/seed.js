/**
 * Seed LibreChat so the UI is usable the moment it comes up: an agent with
 * every ASCENT MCP tool already attached.
 *
 * Without this a new user has to build an agent and attach two MCP servers by
 * hand before they can ask anything -- most of the way to just configuring an
 * MCP client instead.
 *
 * Written in Node rather than shell because the LibreChat image ships no curl.
 * Idempotent: finds an existing agent by name and exits.
 *
 * Never fails the stack. Every error path exits 0 -- a missing agent is a
 * degraded demo, not a broken one, and this service blocks nothing.
 */

const fs = require('fs');

const BASE = process.env.LIBRECHAT_URL || 'http://librechat:3080';
const EMAIL = process.env.CHAT_USER || 'demo@ascent.local';
const PASSWORD = process.env.CHAT_PASSWORD || 'ascentdemo';
const AGENT_NAME = process.env.AGENT_NAME || 'ASCENT';
const INSTRUCTIONS = '/seed/agent-instructions.md';

// LibreChat rejects any request whose User-Agent is not a recognised browser
// (api/server/middleware/uaParser.js) with "Illegal request" -- and returns it
// with HTTP 200, so it does not look like a failure. Every call needs this.
const UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function call(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { 'User-Agent': UA, 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body };
}

async function waitForLibreChat(attempts = 90) {
  for (let i = 0; i < attempts; i++) {
    try {
      const { status } = await call('/api/config');
      if (status === 200) return true;
    } catch {
      /* not listening yet */
    }
    await sleep(2000);
  }
  return false;
}

const SKILLS_DIR = '/seed-skills';

/** Split a SKILL.md into its YAML frontmatter and body. */
function parseSkill(text) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text.trimStart());
  if (!match) return { frontmatter: {}, body: text };

  const frontmatter = {};
  let key = null;
  for (const line of match[1].split(/\r?\n/)) {
    const pair = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
    if (pair) {
      key = pair[1];
      frontmatter[key] = pair[2].trim().replace(/^["']|["']$/g, '');
    } else if (key && line.trim()) {
      // A folded value continued on the next line, which `description:` often is.
      frontmatter[key] = `${frontmatter[key]} ${line.trim()}`.trim();
    }
  }
  return { frontmatter, body: match[2] };
}

async function seedSkills(auth) {
  let entries;
  try {
    entries = fs.readdirSync(SKILLS_DIR, { withFileTypes: true }).filter((e) => e.isDirectory());
  } catch {
    console.log('seed: no skills directory mounted; skipping skills');
    return;
  }

  const listed = await call('/api/skills', { headers: auth });
  const already = new Set(((listed.body && listed.body.skills) || []).map((s) => s.name));

  for (const entry of entries) {
    const path = `${SKILLS_DIR}/${entry.name}/SKILL.md`;
    if (!fs.existsSync(path)) continue;

    const { frontmatter, body } = parseSkill(fs.readFileSync(path, 'utf8'));
    // LibreChat requires kebab-case; the directory name already is one, and is
    // the identifier Claude Code triggers on, so prefer it over frontmatter.
    const name = entry.name;
    const description = frontmatter.description;
    if (!description) {
      console.log(`seed: ${name} has no description; skipping`);
      continue;
    }
    if (already.has(name)) {
      console.log(`seed: skill ${name} already exists`);
      continue;
    }

    const created = await call('/api/skills', {
      method: 'POST',
      headers: auth,
      body: JSON.stringify({ name, description, body, frontmatter }),
    });
    if (created.status === 200 || created.status === 201) {
      console.log(`seed: created skill ${name}`);
      await sharePublicly('skill', created.body && created.body._id, 'skill_viewer', name, auth);
    } else {
      console.log(`seed: skill ${name} failed HTTP ${created.status}:`, JSON.stringify(created.body).slice(0, 200));
    }
  }
}

/**
 * Grant everyone read access to a seeded resource.
 *
 * Sharing is an ACL, not a field: `isPublic: true` on the create call is
 * silently dropped -- the skill is stored, the flag is not, and the seed
 * reports success. Everything then works for the seeding account and is
 * invisible to every other user, which reads as "the skills did not install"
 * rather than "they are not yours". That is exactly what the deployed stack
 * did: the skills were created, none visible to the Cognito user who signed in.
 */
async function sharePublicly(resourceType, id, roleId, label, auth) {
  if (!id) {
    console.log(`seed: ${resourceType} ${label} has no id; cannot share`);
    return;
  }
  const res = await call(`/api/permissions/${resourceType}/${id}`, {
    method: 'PUT',
    headers: auth,
    body: JSON.stringify({ updated: [], removed: [], public: true, publicAccessRoleId: roleId }),
  });
  if (res.status === 200) {
    console.log(`seed: shared ${resourceType} ${label} publicly`);
  } else {
    console.log(`seed: share ${resourceType} ${label} failed HTTP ${res.status}:`, JSON.stringify(res.body).slice(0, 200));
  }
}

async function main() {
  if (!(await waitForLibreChat())) {
    console.log('seed: LibreChat did not come up; skipping agent');
    return;
  }

  const login = await call('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email: EMAIL, password: PASSWORD }),
  });
  const token = login.body && login.body.token;
  if (!token) {
    console.log(`seed: could not log in (HTTP ${login.status}); skipping agent`);
    return;
  }
  const auth = { Authorization: `Bearer ${token}` };

  await seedSkills(auth);

  const existing = await call('/api/agents', { headers: auth });
  const list = Array.isArray(existing.body) ? existing.body : existing.body?.data || [];
  if (list.some((a) => a && a.name === AGENT_NAME)) {
    console.log(`seed: agent ${AGENT_NAME} already exists`);
    return;
  }

  // Tool identifiers are assigned by LibreChat as {tool}_mcp_{server}, so they
  // are read back rather than hardcoded -- renaming a server in librechat.yaml
  // would otherwise leave the agent holding names that no longer resolve.
  // LibreChat answers /api/config before it has finished connecting to its MCP
  // servers, so asking once races startup: on a warm instance the tools are
  // already there, on a genuine first run they are not, and the agent is
  // created with nothing attached -- or skipped entirely.
  let tools = [];
  for (let i = 0; i < 60; i++) {
    const mcp = await call('/api/mcp/tools', { headers: auth });
    const servers = (mcp.body && mcp.body.servers) || {};
    tools = Object.values(servers).flatMap((s) => (s.tools || []).map((t) => t.pluginKey));
    if (tools.length) break;
    await sleep(5000);
  }
  if (!tools.length) {
    console.log('seed: MCP tools never registered; skipping agent');
    return;
  }

  const created = await call('/api/agents', {
    method: 'POST',
    headers: auth,
    body: JSON.stringify({
      name: AGENT_NAME,
      description: 'Epidemiological questions over synthetic real-world health data.',
      instructions: fs.readFileSync(INSTRUCTIONS, 'utf8'),
      provider: process.env.AGENT_PROVIDER || 'google',
      // The one model the picker offers. Left at 2.5-flash the agent quietly
      // ran a model that is no longer in GOOGLE_MODELS -- the picker showed
      // 3.7 and the agent answered on something else.
      model: process.env.AGENT_MODEL || 'gemini-3.7-flash',
      tools,
      // "Use all skills": enabled with an empty allowlist. Without it the agent
      // is created with the skills section switched off, so the skills
      // seeded just above are installed and unreachable.
      skills_enabled: true,
      skills: [],
    }),
  });

  if (created.status === 201 || created.status === 200) {
    console.log(`seed: created agent ${AGENT_NAME} with ${tools.length} tools`);
    // _id, not id: the permissions API keys off the Mongo document id and
    // rejects the agent_* identifier with "Invalid resource ID".
    await sharePublicly('agent', created.body && (created.body._id || created.body.id), 'agent_viewer', AGENT_NAME, auth);
  } else {
    console.log(`seed: agent create failed HTTP ${created.status}:`, created.body);
  }
}

main().catch((error) => console.log('seed: skipped —', error.message));
