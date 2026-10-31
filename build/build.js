// Gridiron Index build: pulls ESPN data, verifies records, renders index.html (GitHub Pages) and artifact.html (Claude artifact).
// Usage: node build/build.js            (fetch live data)
//        node build/build.js --data f   (render from a saved data file, no network)
const fs = require('fs'), path = require('path');
const { rankModel } = require('./model.js');
const ROOT = path.join(__dirname, '..');
const B = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football';
const C = 'https://sports.core.api.espn.com/v2/sports/football/leagues/college-football';
const CONF = { 8: 'SEC', 151: 'AAC', 4: 'B12', 15: 'MAC', 9: 'P12', 17: 'MWC', 1: 'ACC', 5: 'B1G', 18: 'IND', 12: 'CUSA', 37: 'SBC' };
const NAMES = require('./names.json');

async function getJSON(url, tries = 4) {
  for (let i = 0; i < tries; i++) {
    try { const r = await fetch(url, { headers: { 'User-Agent': 'gridiron-index' } }); if (r.ok) return await r.json(); if (r.status === 404) return null; }
    catch (e) { /* retry */ }
    await new Promise(res => setTimeout(res, 1000 * (i + 1)));
  }
  throw new Error('Failed: ' + url);
}
async function pool(items, fn, n = 10) {
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) { const x = items[i++]; await fn(x); } }));
}
function seasonStart(yr) { // Monday on or before Aug 25
  const d = new Date(Date.UTC(yr, 7, 25)); const dow = d.getUTCDay(); d.setUTCDate(d.getUTCDate() - ((dow + 6) % 7)); return d.getTime();
}

async function fetchData() {
  const now = new Date();
  const season = now.getUTCMonth() >= 6 ? now.getUTCFullYear() : now.getUTCFullYear() - 1;
  const t0 = seasonStart(season);
  const list = await getJSON(`${C}/seasons/${season}/types/2/groups/80/teams?limit=300`);
  const ids = list.items.map(x => x.$ref.match(/teams\/(\d+)/)[1]);
  const T = {}, G = {};
  const jobs = []; ids.forEach(id => { jobs.push([id, 2]); jobs.push([id, 3]); });
  await pool(jobs, async ([id, st]) => {
    const s = await getJSON(`${B}/teams/${id}/schedule?season=${season}&seasontype=${st}`);
    if (!s) return;
    if (st === 2) { const gr = s.team?.groups || {}; T[id] = { abbr: s.team.abbreviation, short: s.team.shortDisplayName, gid: gr.id, pid: gr.parent?.id, rec: s.team.recordSummary || '' }; }
    for (const e of s.events || []) {
      const c = e.competitions[0]; if (!c.status.type.completed) continue;
      const sc = x => x.score?.value ?? +x.score;
      const h = c.competitors.find(x => x.homeAway === 'home') || c.competitors[0], a = c.competitors.find(x => x !== h);
      G[e.id] = { ev: e.id, date: e.date, h: h.team.id, hs: sc(h), a: a.team.id, as: sc(a), neu: c.neutralSite ? 1 : 0 };
    }
  });
  const fbs = Object.entries(T).filter(([, v]) => v.gid).sort((x, y) => (x[1].abbr < y[1].abbr ? -1 : 1));
  if (fbs.length < 120) throw new Error('Only ' + fbs.length + ' FBS teams found');
  const idx = {}; fbs.forEach(([id], i) => (idx[id] = i));
  const games = Object.values(G).filter(g => g.h in idx || g.a in idx);
  await pool(games, async g => {
    const sm = await getJSON(`${B}/summary?event=${g.ev}`);
    const p = (sm?.pickcenter || [])[0];
    g.line = p && typeof p.spread === 'number' ? p.spread : null;
    if (g.line === null) {
      const o = await getJSON(`${C}/events/${g.ev}/competitions/${g.ev}/odds`).catch(() => null);
      const it = (o?.items || []).filter(x => !/live/i.test(x.provider?.name || '') && typeof x.spread === 'number');
      if (it[0]) g.line = it[0].spread;
    }
    for (const t of sm?.boxscore?.teams || []) {
      const m = {}; for (const s of t.statistics) m[s.name] = s.displayValue;
      const pa = +(m.completionAttempts || '0/0').split('/')[1], ra = +m.rushingAttempts, ty = +m.totalYards;
      const y = pa + ra > 0 && isFinite(ty) ? ty / (pa + ra) : null;
      if (t.team.id === g.h) g.hy = y; else if (t.team.id === g.a) g.ay = y;
    }
  });
  const conf = ([, v]) => { const k = v.pid == null || Number(v.pid) === 80 ? Number(v.gid) : Number(v.pid); return CONF[k] || 'G' + k; };
  const D = {
    season, t0,
    teams: fbs.map(([, v]) => v.abbr),
    names: fbs.map(([, v]) => NAMES[v.abbr] || v.short),
    conf: fbs.map(conf),
    records: fbs.map(([, v]) => v.rec),
    games: games.sort((x, y) => (x.date < y.date ? -1 : x.date > y.date ? 1 : x.ev < y.ev ? -1 : 1)).map(g => [
      Math.floor((Date.parse(g.date) - t0) / 864e5 / 7),
      g.h in idx ? idx[g.h] : null, g.hs, g.a in idx ? idx[g.a] : null, g.as,
      g.line == null ? 0 : g.line, g.neu, g.line == null ? 1 : 0,
      g.hy == null ? null : +g.hy.toFixed(3), g.ay == null ? null : +g.ay.toFixed(3)]),
    lastDate: games.reduce((m, g) => (g.date > m ? g.date : m), ''),
  };
  D.polls = await fetchPolls(idx);
  return D;
}

// AP poll and College Football Playoff committee rankings from ESPN. Optional: a failure here never blocks the build.
// Each poll: { key, label, name, date, ranks } where ranks[i] is team i's rank (or null if unranked).
async function fetchPolls(idx) {
  try {
    const r = await getJSON(`${B}/rankings`);
    const want = [
      ['ap', 'AP', p => p.type === 'ap' || /^AP\b/i.test(p.name || '')],
      ['cfp', 'CFP', p => p.type === 'cfp' || /playoff|\bCFP\b/i.test(`${p.name} ${p.shortName}`)],
    ];
    const out = [];
    for (const [key, label, match] of want) {
      const p = (r?.rankings || []).find(match);
      if (!p || !(p.ranks || []).length) { console.log(`Poll ${label}: not available`); continue; }
      const ranks = Object.keys(idx).map(() => null);
      let hit = 0;
      for (const x of p.ranks) { const i = idx[x.team?.id]; if (i != null && x.current) { ranks[i] = x.current; hit++; } }
      console.log(`Poll ${label}: ${p.name}, ${p.date}, ${hit}/${p.ranks.length} teams matched`);
      if (hit < 20) { console.warn(`Poll ${label}: too few teams matched, skipping`); continue; }
      out.push({ key, label, name: p.name, date: p.date || p.ranks[0]?.date || '', ranks });
    }
    return out;
  } catch (e) { console.warn('Polls unavailable:', e.message); return []; }
}

function verify(D) {
  const wl = D.teams.map(() => [0, 0]);
  for (const [, h, hs, a, as] of D.games) { if (h !== null) wl[h][hs > as ? 0 : 1]++; if (a !== null) wl[a][as > hs ? 0 : 1]++; }
  const bad = D.teams.map((t, i) => [t, wl[i].join('-'), D.records[i]]).filter(([, mine, off]) => off && mine !== off);
  const noYpp = D.games.filter(g => g[1] !== null && g[3] !== null && (g[8] == null || g[9] == null)).length;
  const unknownConf = [...new Set(D.conf.filter(c => !Object.values(CONF).includes(c)))];
  const confSizes = {}; D.conf.forEach(c => (confSizes[c] = (confSizes[c] || 0) + 1));
  if (unknownConf.length || Object.keys(confSizes).length < 8) bad.push(['conferences', JSON.stringify(confSizes), 'expected known FBS conferences']);
  return { bad, noYpp };
}

const fmtET = (d, opts) => new Date(d).toLocaleString('en-US', { timeZone: 'America/New_York', ...opts });
// Header line naming the poll dates shown in the table. The CFP committee's first ranking comes out the first Tuesday of November.
function pollMeta(polls, season) {
  const day = d => (d ? fmtET(d, { month: 'short', day: 'numeric' }) : '');
  const parts = polls.map(p => `<span>${p.label} = ${p.key === 'ap' ? 'AP poll' : 'CFP committee'} of <b>${day(p.date)}</b></span>`);
  if (!polls.some(p => p.key === 'cfp')) {
    const t = new Date(Date.UTC(season, 10, 1)); t.setUTCDate(1 + ((9 - t.getUTCDay()) % 7)); // first Tuesday of November
    parts.push(`<span>CFP = committee rankings, first release <b>${t.toLocaleString('en-US', { timeZone: 'UTC', weekday: 'short', month: 'short', day: 'numeric' })}</b></span>`);
  }
  return parts.join('');
}
function render(D) {
  const curWeek = Math.max(...D.games.map(g => g[0]));
  const prevWeek = curWeek - 1;
  const prevEnd = D.t0 + (prevWeek * 7 + 6) * 864e5 + 12 * 3600e3; // Sunday of the previous week
  const meta = {
    __SEASON__: String(D.season),
    __NTEAMS__: String(D.teams.length),
    __NGAMES__: String(D.games.length),
    __ASOF__: fmtET(D.lastDate, { weekday: 'short', month: 'short', day: 'numeric' }),
    __PREVTHRU__: fmtET(prevEnd, { timeZone: 'UTC', weekday: 'short', month: 'short', day: 'numeric' }),
    __UPDATED__: fmtET(Date.now(), { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) + ' ET',
    __PREVWEEK__: String(prevWeek),
    __POLLMETA__: pollMeta(D.polls || [], D.season),
  };
  let page = fs.readFileSync(path.join(__dirname, 'page.html'), 'utf8');
  const data = { teams: D.teams, names: D.names, conf: D.conf, games: D.games, polls: (D.polls || []).map(p => ({ key: p.key, label: p.label, name: p.name, ranks: p.ranks })) };
  page = page.replace('__DATA__', () => JSON.stringify(data)).replace('__MODEL__', () => fs.readFileSync(path.join(__dirname, 'model.js'), 'utf8'));
  for (const [k, v] of Object.entries(meta)) page = page.split(k).join(v);
  if (/__[A-Z]+__/.test(page)) throw new Error('Unfilled placeholder: ' + page.match(/__[A-Z]+__/)[0]);
  const cut = page.indexOf('</style>') + '</style>'.length;
  const head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' +
    '<meta name="description" content="Bob\'s Top 25: a college football FBS Top 25 adjusted for strength of schedule, yards per play, conference strength and performance against the spread.">\n' +
    '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n';
  const index = head + page.slice(0, cut).replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '') + '\n</head>\n<body>\n' + page.slice(cut) + '\n</body>\n</html>\n';
  fs.writeFileSync(path.join(ROOT, 'index.html'), index);
  fs.writeFileSync(path.join(ROOT, 'artifact.html'), page);
  const top = rankModel(data, 99).sort((a, b) => a.rank - b.rank).slice(0, 5).map(r => `${r.rank}. ${r.name} ${r.w}-${r.l}`);
  return { meta, top };
}

(async () => {
  const i = process.argv.indexOf('--data');
  const D = i > 0 ? JSON.parse(fs.readFileSync(process.argv[i + 1], 'utf8')) : await fetchData();
  const v = verify(D);
  console.log(`Teams ${D.teams.length}, games ${D.games.length}, last game ${D.lastDate}, games missing yards/play ${v.noYpp}`);
  if (v.bad.length) { console.error('Check failed (records or conferences), not publishing:', JSON.stringify(v.bad.slice(0, 10))); process.exit(1); }
  if (i < 0) { fs.mkdirSync(path.join(ROOT, 'data'), { recursive: true }); fs.writeFileSync(path.join(ROOT, 'data', 'latest.json'), JSON.stringify(D)); }
  const { meta, top } = render(D);
  console.log(JSON.stringify(meta)); console.log(top.join('\n'));
})().catch(e => { console.error(e); process.exit(1); });
