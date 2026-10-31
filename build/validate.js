// Walk-forward check of the predictive ratings: for each week w, fit on games before w and predict week w.
// Usage: node build/validate.js [data/latest.json] [--grid]
const fs = require('fs');
const { rate, predict, Phi, P_DEFAULTS } = require('./predict.js');
const { rankModel } = require('./model.js');
const file = process.argv.find(a => a.endsWith('.json')) || __dirname + '/../data/latest.json';
const D = JSON.parse(fs.readFileSync(file, 'utf8'));
const lastW = Math.max(...D.games.map(g => g[0]));
const START = 4; // weeks 0-3 are too early to rate teams without a preseason prior

function score(o) {
  let n = 0, ll = 0, br = 0, hit = 0, se = 0, nm = 0, llm = 0, brm = 0, hitm = 0, sem = 0, hitPoll = 0;
  for (let w = START; w <= lastW; w++) {
    const R = rate(D, w - 1, o);
    const poll = rankModel(D, w - 1);
    for (const [gw, h, hs, a, as, line, neu, noLine] of D.games) {
      if (gw !== w || h === null || a === null || hs === as) continue;
      const { m } = predict(R, h, a, neu, o), p = Phi(m / o.sigma), y = hs > as ? 1 : 0, act = hs - as;
      n++; ll -= Math.log(y ? p : 1 - p); br += (p - y) ** 2; hit += (p > 0.5) === !!y; se += (m - act) ** 2;
      hitPoll += (poll[h].score + (neu ? 0 : 0) > poll[a].score) === !!y;
      if (!noLine) { const pm = Phi(-line / o.sigma); nm++; llm -= Math.log(y ? pm : 1 - pm); brm += (pm - y) ** 2; hitm += (line < 0) === !!y || (line === 0 && false); sem += (-line - act) ** 2; }
    }
  }
  return { n, logLoss: ll / n, brier: br / n, acc: hit / n, rmse: Math.sqrt(se / n), pollAcc: hitPoll / n, nMkt: nm, mktLogLoss: llm / nm, mktBrier: brm / nm, mktAcc: hitm / nm, mktRmse: Math.sqrt(sem / nm) };
}

const f3 = x => x.toFixed(3);
if (process.argv.includes('--grid')) {
  const out = [];
  for (const cap of [21, 28]) for (const ridge of [0.5, 1, 2, 3]) for (const wPts of [0.25, 0.5, 0.75]) for (const wMkt of [0, 0.25, 0.5, 0.75, 0.9, 1]) {
    const o = { ...P_DEFAULTS, cap, ridge, wPts, wMkt, sigma: 15 };
    const s = score(o); out.push([s.brier, o, s]);
  }
  out.sort((x, y) => x[0] - y[0]);
  for (const [, o, s] of out.slice(0, 12)) console.log(`cap ${o.cap} ridge ${o.ridge} wPts ${o.wPts} wMkt ${o.wMkt}: brier ${f3(s.brier)} logloss ${f3(s.logLoss)} acc ${f3(s.acc)} rmse ${s.rmse.toFixed(1)}`);
  for (const wMkt of [0, 0.5, 1]) { const b = out.find(x => x[1].wMkt === wMkt); console.log(`best with wMkt ${wMkt}: brier ${f3(b[2].brier)} acc ${f3(b[2].acc)}`); }
  const best = out[0][1];
  for (const sigma of [12, 13, 14, 15, 16, 17, 18]) { const s = score({ ...best, sigma }); console.log(`  sigma ${sigma}: logloss ${f3(s.logLoss)} brier ${f3(s.brier)}`); }
} else {
  const s = score(P_DEFAULTS);
  console.log(`Weeks ${START}-${lastW}, ${s.n} FBS-vs-FBS games, fit only on earlier weeks`);
  console.log(`Ratings:   picks ${(100 * s.acc).toFixed(1)}%  brier ${f3(s.brier)}  logloss ${f3(s.logLoss)}  margin error ${s.rmse.toFixed(1)} pts`);
  console.log(`Reactive poll picks ${(100 * s.pollAcc).toFixed(1)}% (higher score wins)`);
  console.log(`ESPN line (${s.nMkt} games): picks ${(100 * s.mktAcc).toFixed(1)}%  brier ${f3(s.mktBrier)}  logloss ${f3(s.mktLogLoss)}  margin error ${s.mktRmse.toFixed(1)} pts`);
}
