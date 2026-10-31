// Gridiron Index ranking model.
// games: [week, homeIdx|null, homePts, awayIdx|null, awayPts, homeSpread, neutral]; null = FCS team.
// homeSpread is ESPN's (DraftKings) line for the home team: negative = home favored.
// v3: v2 weights plus opponent-adjusted yards per play (0.30 raw = 23% after normalizing). Backtested on 2024-25.
const DEFAULT_W = { wp: 0.27, owp: 0.19, oowp: 0.08, conf: 0.08, ypp: 0.23, ats: 0.08, mov: 0.04, qual: 0.03 }; // sums to 1; rounded from the tested mix
const DEFAULT_FCS_WIN = 0.4;               // an FCS win counts as 0.4 of a game won (losses count in full)
const FCS_WP = 0.30, FCS_CONF = 0.20, MOV_CAP = 24, ATS_CAP = 21, CONF_PRIOR = 4;

function rankModel(D, maxWeek, W, fcsWin, X) {
  W = W || DEFAULT_W; fcsWin = fcsWin == null ? DEFAULT_FCS_WIN : fcsWin; X = X || { yppAdj: 1 };
  // X.loc: road wins / home losses weigh (1+loc), home wins / road losses (1-loc); neutral 1
  // X.close: games decided by <= 7 pts weigh close (e.g. 0.5) in win %
  // X.half: recency half-life in weeks (0 = off), applied to win %, margin and ATS
  const n = D.teams.length;
  const sched = Array.from({ length: n }, () => []);
  let lastW = 0;
  for (const [w, h, hs, a, as, line, neu, noLine, hy, ay] of D.games) {
    if (w > maxWeek) continue; lastW = Math.max(lastW, w);
    if (h !== null) sched[h].push({ opp: a, pf: hs, pa: as, line: line, home: true, neu: !!neu, w, yf: hy, ya: ay });
    if (a !== null) sched[a].push({ opp: h, pf: as, pa: hs, line: -line, home: false, neu: !!neu, w, yf: ay, ya: hy });
  }
  const win = g => g.pf > g.pa;
  // weight of a game in the team's own results: FCS wins are discounted, everything else counts in full
  const rec = g => (X.half ? Math.pow(0.5, (lastW - g.w) / X.half) : 1);
  const gw = g => (g.opp === null && win(g) ? fcsWin : 1) * rec(g);
  const loc = g => (!X.loc || g.neu ? 1 : (win(g) !== g.home ? 1 + X.loc : 1 - X.loc));
  const cls = g => (X.close != null && Math.abs(g.pf - g.pa) <= 7 ? X.close : 1);
  const gwWP = g => gw(g) * loc(g) * cls(g);
  const wl = sched.map(s => { let w = 0, l = 0; for (const g of s) win(g) ? w++ : l++; return [w, l]; });
  const pct = (w, l) => (w + l ? w / (w + l) : 0);
  const WP = sched.map(s => { let num = 0, den = 0; for (const g of s) { const k = gwWP(g); den += k; if (win(g)) num += k; } return den ? num / den : 0; });
  const rawWP = wl.map(([w, l]) => pct(w, l));
  const wpExcl = (o, t) => { let [w, l] = wl[o]; for (const g of sched[o]) if (g.opp === t) win(g) ? w-- : l--; return pct(w, l); };
  const OWP = sched.map((s, t) => avg(s.map(g => g.opp === null ? FCS_WP : wpExcl(g.opp, t))));
  const OOWP = sched.map(s => avg(s.map(g => g.opp === null ? FCS_WP : OWP[g.opp])));
  const cw = {}, cl = {};
  for (let t = 0; t < n; t++) for (const g of sched[t]) {
    if (g.opp === null || D.conf[g.opp] === D.conf[t]) continue;
    const c = D.conf[t]; cw[c] = cw[c] || 0; cl[c] = cl[c] || 0; win(g) ? cw[c]++ : cl[c]++;
  }
  const CS = {};
  for (const c of new Set(D.conf)) CS[c] = c === 'IND' ? 0.5 : ((cw[c] || 0) + CONF_PRIOR / 2) / ((cw[c] || 0) + (cl[c] || 0) + CONF_PRIOR);
  const OCONF = sched.map(s => avg(s.map(g => g.opp === null ? FCS_CONF : CS[D.conf[g.opp]])));
  const clip = (x, c) => Math.max(-c, Math.min(c, x));
  const MOV = sched.map(s => (wavg(s.map(g => [clip(g.pf - g.pa, MOV_CAP) / MOV_CAP, gw(g)])) + 1) / 2);
  // Against the spread: actual margin + own line. +3 means beat the number by 3.
  const atsM = g => g.pf - g.pa + g.line;
  const ATS = sched.map(s => (wavg(s.map(g => [clip(atsM(g), ATS_CAP) / ATS_CAP, gw(g)])) + 1) / 2);
  const ATSR = sched.map(s => { let c = 0, x = 0, p = 0, m = 0; for (const g of s) { const v = atsM(g); m += v; v > 0 ? c++ : v < 0 ? x++ : p++; } return { c, x, p, avg: s.length ? m / s.length : 0 }; });
  // Yards per play: X.yppAdj compares each game to what the opponent normally allows / gains (FBS games only)
  const hasY = g => g.yf != null && g.ya != null;
  const offA = sched.map(s => avg(s.filter(g => hasY(g) && g.opp !== null).map(g => g.yf)));
  const defA = sched.map(s => avg(s.filter(g => hasY(g) && g.opp !== null).map(g => g.ya)));
  const YPP = sched.map(s => (wavg(s.filter(hasY).map(g => {
    if (X.yppAdj) { if (g.opp === null) return [0, 0]; return [clip((g.yf - defA[g.opp]) + (offA[g.opp] - g.ya), 4) / 4, rec(g)]; }
    return [clip(g.yf - g.ya, 3) / 3, gw(g)];
  })) + 1) / 2);
  const keys = ['wp', 'owp', 'oowp', 'conf', 'ats', 'mov', 'ypp'].filter(k => W[k] != null);
  const comp = t => ({ wp: WP[t], owp: OWP[t], oowp: OOWP[t], conf: OCONF[t], ats: ATS[t], mov: MOV[t], ypp: YPP[t] });
  const noQ = keys.reduce((s, k) => s + W[k], 0) || 1;
  const base = WP.map((_, t) => { const c = comp(t); return keys.reduce((s, k) => s + W[k] * c[k], 0) / noQ; });
  const baseRank = ranks(base);
  const QG = sched.map(s => {
    let q = 0, wsum = 0, top25 = 0, bad = 0;
    for (const g of s) {
      const r = g.opp === null ? 999 : baseRank[g.opp], k = gw(g); wsum += k;
      if (win(g)) { q += k * (r <= 10 ? 1 : r <= 25 ? 0.75 : r <= 50 ? 0.4 : 0); if (r <= 25) top25++; }
      else { const v = r === 999 ? -1 : r > 75 ? -0.6 : r > 50 ? -0.3 : 0; q += v; if (v < 0) bad++; }
    }
    return { q: wsum ? q / wsum : 0, top25, bad };
  });
  const QUAL = QG.map(x => Math.max(0, Math.min(1, (x.q + 1) / 2)));
  const score = WP.map((_, t) => { const c = comp(t); return keys.reduce((s, k) => s + W[k] * c[k], 0) + W.qual * QUAL[t]; });
  const rank = ranks(score);
  const sosRank = ranks(OWP.map((o, t) => (2 * o + OOWP[t]) / 3));
  const wlRank = ranks(wl.map(([w], t) => rawWP[t] * 1000 + w), true);
  const atsRank = ranks(ATSR.map(r => r.avg));
  return D.teams.map((abbr, t) => ({
    t, abbr, name: D.names[t], conf: D.conf[t], w: wl[t][0], l: wl[t][1],
    fcsW: sched[t].filter(g => g.opp === null && win(g)).length,
    wp: WP[t], rawWP: rawWP[t], owp: OWP[t], oowp: OOWP[t], oconf: OCONF[t], mov: MOV[t], ats: ATS[t], atsRec: ATSR[t],
    ypp: YPP[t], yOff: offA[t], yDef: defA[t], qual: QUAL[t], top25: QG[t].top25, bad: QG[t].bad, score: score[t],
    rank: rank[t], sosRank: sosRank[t], wlRank: wlRank[t], atsRank: atsRank[t]
  }));
}
function avg(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }
function wavg(a) { let s = 0, w = 0; for (const [x, k] of a) { s += x * k; w += k; } return w ? s / w : 0; }
function ranks(v, ties) {
  const idx = v.map((_, i) => i).sort((a, b) => v[b] - v[a]);
  const r = new Array(v.length);
  idx.forEach((i, k) => { r[i] = ties ? 1 + v.filter(x => x > v[i] + 1e-12).length : k + 1; });
  return r;
}
function confStrength(D, maxWeek) {
  const rec = {};
  for (const [w, h, hs, a, as] of D.games) {
    if (w > maxWeek || a === null || h === null || D.conf[a] === D.conf[h]) continue;
    for (const [x, won] of [[h, hs > as], [a, as > hs]]) { const c = D.conf[x]; rec[c] = rec[c] || [0, 0]; rec[c][won ? 0 : 1]++; }
  }
  return rec;
}
if (typeof module !== 'undefined') module.exports = { rankModel, confStrength, DEFAULT_W, DEFAULT_FCS_WIN };
