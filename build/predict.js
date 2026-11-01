// Predictive side of Bob's Top 25: point ratings, game win probabilities and a Monte Carlo of the rest of the season.
// Used by build.js (node). The page only shows the precomputed results.
const { rankModel } = require('./model.js');

// Tuned on 2026 walk-forward predictions (see build/validate.js and the project's predictive-model doc).
const P_DEFAULTS = {
  cap: 28,        // game margins are capped at +-28 before fitting, so blowouts don't dominate
  ridge: 0.5,     // light shrink toward average (half a game at 0) so teams with few games stay sensible
  wPts: 0.25,     // performance blend: 25% scoring-margin rating, 75% yards-per-play rating (converted to points)
  wMkt: 0.5,      // final blend: share given to the market rating implied by past ESPN point spreads
  sigma: 15,      // spread of actual margins around the predicted margin, in points
  tau: 3.5,       // uncertainty in each team's rating (part of sigma), shared across that team's games in a simulation
};

function solve(A, b) { // Gaussian elimination with partial pivoting; A is modified
  const n = b.length;
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < n; r++) { const f = A[r][c] / A[c][c]; if (!f) continue; for (let k = c; k < n; k++) A[r][k] -= f * A[c][k]; b[r] -= f * b[c]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = b[r]; for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k]; x[r] = s / A[r][r]; }
  return x;
}

// Least-squares team ratings: y = r[home] - r[away] + hfa (if not neutral). All FCS teams share one rating (index n).
function fit(n, rows, ridge) {
  const p = n + 2, H = n + 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
  for (const [h, a, neu, y] of rows) {
    const x = [[h, 1], [a, -1]]; if (!neu) x.push([H, 1]);
    for (const [i, vi] of x) { b[i] += vi * y; for (const [j, vj] of x) A[i][j] += vi * vj; }
  }
  for (let i = 0; i <= n; i++) A[i][i] += ridge;
  A[H][H] += 1e-6;
  const s = solve(A, b);
  return { r: s.slice(0, n + 1), hfa: s[H] };
}

// Ratings from completed games up to maxWeek. Returns points-per-game ratings vs an average FBS team.
function rate(D, maxWeek = 99, o = P_DEFAULTS) {
  const n = D.teams.length, ix = t => (t === null ? n : t);
  const g = D.games.filter(x => x[0] <= maxWeek);
  const clip = v => Math.max(-o.cap, Math.min(o.cap, v));
  const pts = fit(n, g.map(([, h, hs, a, as, , neu]) => [ix(h), ix(a), neu, clip(hs - as)]), o.ridge);
  const yrows = g.filter(x => x[8] != null && x[9] != null).map(([, h, , a, , , neu, , hy, ay]) => [ix(h), ix(a), neu, hy - ay]);
  const ypp = fit(n, yrows, o.ridge);
  // Convert the yards-per-play rating to points: regress capped margin on the predicted ypp edge
  let sxy = 0, sxx = 0;
  for (const [, h, hs, a, as, , neu, , hy, ay] of g) { if (hy == null || ay == null) continue; const e = ypp.r[ix(h)] - ypp.r[ix(a)] + (neu ? 0 : ypp.hfa); sxy += e * clip(hs - as); sxx += e * e; }
  const k = sxx ? sxy / sxx : 7;
  let r = pts.r.map((v, i) => o.wPts * v + (1 - o.wPts) * k * ypp.r[i]);
  let hfa = o.wPts * pts.hfa + (1 - o.wPts) * k * ypp.hfa;
  // Market rating: the team strengths that best explain the pregame point spreads of games already played
  const mrows = g.filter(x => !x[7]).map(([, h, , a, , line, neu]) => [ix(h), ix(a), neu, -line]);
  if (o.wMkt > 0 && mrows.length) {
    const mk = fit(n, mrows, o.ridge);
    r = r.map((v, i) => (1 - o.wMkt) * v + o.wMkt * mk.r[i]);
    hfa = (1 - o.wMkt) * hfa + o.wMkt * mk.hfa;
  }
  // Center on the FBS average so 0 = average FBS team
  const avg = r.slice(0, n).reduce((s, v) => s + v, 0) / n;
  return { r: r.map(v => v - avg), hfa, k, n };
}

const erf = x => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
// Predicted home margin and home win probability for one game
function predict(R, h, a, neu, o = P_DEFAULTS) {
  const n = R.n, m = R.r[h === null ? n : h] - R.r[a === null ? n : a] + (neu ? 0 : R.hfa);
  return { m, p: Phi(m / o.sigma) };
}

function mulberry32(seed) { return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(rnd) { let u = 0; while (!u) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); }

// Simulate the remaining regular season nSims times; each finished season is ranked by the reactive model.
function simulate(D, nSims = 2000, o = P_DEFAULTS, seed = 20261) {
  const R = rate(D, 99, o), n = D.teams.length, rnd = mulberry32(seed);
  const fut = D.future || [];
  // Per-team averages used to fill in plausible scores and yards per play for simulated games
  const acc = Array.from({ length: n + 1 }, () => ({ pf: 0, pa: 0, yf: 0, ya: 0, g: 0, gy: 0 }));
  for (const [, h, hs, a, as, , , , hy, ay] of D.games) for (const [t, f, ag, yf, ya] of [[h, hs, as, hy, ay], [a, as, hs, ay, hy]]) {
    const s = acc[t === null ? n : t]; s.pf += f; s.pa += ag; s.g++; if (yf != null && ya != null) { s.yf += yf; s.ya += ya; s.gy++; }
  }
  const av = acc.map(s => ({ pf: s.g ? s.pf / s.g : 28, pa: s.g ? s.pa / s.g : 28, yf: s.gy ? s.yf / s.gy : 5.6, ya: s.gy ? s.ya / s.gy : 5.6 }));
  // Yards-per-play edge per point of margin, from completed games
  let sxy = 0, sxx = 0; for (const g of D.games) if (g[8] != null && g[9] != null) { const m = g[2] - g[4]; sxy += m * (g[8] - g[9]); sxx += m * m; }
  const yPerPt = sxx ? sxy / sxx : 0.08;
  const sg = Math.sqrt(Math.max(1, o.sigma * o.sigma - 2 * o.tau * o.tau));
  const base = fut.map(([w, h, a, neu, line]) => { const pr = predict(R, h, a, neu, o); return { w, h, a, neu, line, mean: line != null ? -line : pr.m, p: line != null ? Phi(-line / o.sigma) : pr.p }; });

  const rankSum = new Float64Array(n), winSum = new Float64Array(n), lossSum = new Float64Array(n), top25 = new Float64Array(n), unbeaten = new Float64Array(n);
  const hist = Array.from({ length: n }, () => new Uint32Array(n + 1));
  const ix = t => (t === null ? n : t);
  const w0 = D.teams.map(() => [0, 0]); for (const [, h, hs, a, as] of D.games) { if (h !== null) w0[h][hs > as ? 0 : 1]++; if (a !== null) w0[a][as > hs ? 0 : 1]++; }
  for (let s = 0; s < nSims; s++) {
    const pert = Array.from({ length: n + 1 }, () => gauss(rnd) * o.tau);
    const games = D.games.slice();
    const wl = w0.map(x => x.slice());
    for (const g of base) {
      const hi = ix(g.h), ai = ix(g.a);
      let m = g.mean + pert[hi] - pert[ai] + gauss(rnd) * sg;
      m = Math.round(m); if (m === 0) m = rnd() < 0.5 ? 3 : -3;
      const T = (av[hi].pf + av[ai].pa + av[ai].pf + av[hi].pa) / 2;
      let hs = Math.round((T + m) / 2), as = hs - m; if (as < 0) { hs -= as; as = 0; } if (hs < 0) { as -= hs; hs = 0; }
      const ey = (m - g.mean) * yPerPt / 2;
      const hy = +((av[hi].yf + av[ai].ya) / 2 + ey).toFixed(3), ay = +((av[ai].yf + av[hi].ya) / 2 - ey).toFixed(3);
      games.push([g.w, g.h, hs, g.a, as, -g.mean, g.neu, 0, hy, ay]);
      if (g.h !== null) wl[g.h][hs > as ? 0 : 1]++; if (g.a !== null) wl[g.a][as > hs ? 0 : 1]++;
    }
    const res = rankModel({ teams: D.teams, names: D.names, conf: D.conf, games }, 99);
    for (const r of res) {
      rankSum[r.t] += r.rank; hist[r.t][r.rank]++;
      if (r.rank <= 25) top25[r.t]++;
      winSum[r.t] += wl[r.t][0]; lossSum[r.t] += wl[r.t][1]; if (!wl[r.t][1]) unbeaten[r.t]++;
    }
  }
  const pct = (hst, q) => { let c = 0; const lim = q * nSims; for (let k = 1; k < hst.length; k++) { c += hst[k]; if (c >= lim) return k; } return hst.length - 1; };
  const avgRank = Array.from(rankSum, v => v / nSims);
  const order = avgRank.map((v, i) => i).sort((x, y) => avgRank[x] - avgRank[y] || R.r[y] - R.r[x]);
  const projRank = new Array(n); order.forEach((t, k) => (projRank[t] = k + 1));
  const ratingRank = new Array(n); R.r.slice(0, n).map((v, i) => i).sort((x, y) => R.r[y] - R.r[x]).forEach((t, k) => (ratingRank[t] = k + 1));
  // Remaining strength of schedule: average rating of opponents still to play (FCS = the shared FCS rating)
  const rem = D.teams.map(() => []);
  for (const g of base) { if (g.h !== null) rem[g.h].push(R.r[ix(g.a)] - (g.neu ? 0 : R.hfa)); if (g.a !== null) rem[g.a].push(R.r[ix(g.h)] + (g.neu ? 0 : R.hfa)); }
  const remAvg = rem.map(x => (x.length ? x.reduce((s, v) => s + v, 0) / x.length : null));
  const remOrder = remAvg.map((v, i) => i).filter(i => remAvg[i] != null).sort((x, y) => remAvg[y] - remAvg[x]);
  const remRank = new Array(n).fill(null); remOrder.forEach((t, k) => (remRank[t] = k + 1));
  const r1 = v => Math.round(v * 10) / 10;
  const out = {
    sims: nSims, sigma: o.sigma, hfa: r1(R.hfa), fcs: r1(R.r[n]),
    teams: D.teams.map((_, t) => ({
      rating: r1(R.r[t]), ratingRank: ratingRank[t], projRank: projRank[t], avgRank: r1(avgRank[t]),
      lo: pct(hist[t], 0.1), hi: pct(hist[t], 0.9), top25: Math.round(1000 * top25[t] / nSims) / 1000,
      w: r1(winSum[t] / nSims), l: r1(lossSum[t] / nSims), unbeaten: Math.round(1000 * unbeaten[t] / nSims) / 1000,
      remSos: remRank[t], left: rem[t].length,
    })),
    // Win probability for the home team in each remaining game, same order as D.future
    future: base.map(g => Math.round(1000 * g.p) / 1000),
  };
  {
    const fut = D.future.map(g => g.slice(0, 5));
    // Game-by-game: the favorite (win chance >= 50%, home team on an exact 50-50) is the projected winner of each game,
    // so every game has exactly one winner and the league's projected records always add up.
    out.pick = out.future.map(ph => (ph >= 0.5 ? 1 : 0)); // 1 = home team wins, 0 = away team
    const won = D.teams.map(() => 0);
    for (const [, h, hs, a, as] of D.games) { if (h !== null && hs > as) won[h]++; if (a !== null && as > hs) won[a]++; }
    fut.forEach((g, i) => { const w = out.pick[i] ? g[1] : g[2]; if (w !== null) won[w]++; });
    out.teams.forEach((x, t) => { x.pw = won[t]; x.left = fut.filter(g => g[1] === t || g[2] === t).length; });
    // Projected finish = the current-Top-25 formula applied to this one season (every favorite wins, by its
    // predicted margin), so a team's projected rank always matches its projected W-L. The simulations above
    // still supply the rank range, Top-25 odds and unbeaten odds.
    const games = D.games.slice();
    base.forEach((g, i) => {
      const hi = ix(g.h), ai = ix(g.a), homeWins = out.pick[i] === 1;
      let m = Math.max(1, Math.round(Math.abs(g.mean))); if (!homeWins) m = -m;
      const T = (av[hi].pf + av[ai].pa + av[ai].pf + av[hi].pa) / 2;
      let hs = Math.round((T + m) / 2), as = hs - m; if (as < 0) { hs -= as; as = 0; } if (hs < 0) { as -= hs; hs = 0; }
      const hy = +((av[hi].yf + av[ai].ya) / 2).toFixed(3), ay = +((av[ai].yf + av[hi].ya) / 2).toFixed(3);
      games.push([g.w, g.h, hs, g.a, as, -g.mean, g.neu, 0, hy, ay]);
    });
    for (const r of rankModel({ teams: D.teams, names: D.names, conf: D.conf, games }, 99)) out.teams[r.t].projRank = r.rank;
  }
  return out;
}


module.exports = { P_DEFAULTS, rate, predict, simulate, Phi };
