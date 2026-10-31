# Bob's Top 25

A college football FBS Top 25 that starts from won-lost record and adjusts for strength of schedule, opponent conference strength, performance against the ESPN spread, margin and quality wins. FCS wins are discounted.

Live page: https://netizen-boba.github.io/NCAA-Football-Top25/

## Updates
Rankings rebuild automatically every Saturday, Sunday and Wednesday morning (GitHub Actions, `.github/workflows/update.yml`).
The build (`build/build.js`) pulls scores, box scores and spreads from ESPN, checks every team's record against
ESPN's official standings, and only publishes if they all match. The table also shows the latest AP poll and College Football Playoff
committee rankings from ESPN for comparison (they do not feed the model); the Wednesday run picks up Sunday's AP poll
and Tuesday night's CFP release. Run it by hand from the Actions tab ("Run workflow").
