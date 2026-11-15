# VPS deployment

Run from the repository root in Windows PowerShell after committing and pushing
the intended changes to GitHub `main`:

```powershell
# Check connection, checkout, tmux pane and running Python process only.
powershell -NoProfile -File scripts/deploy_vps.ps1

# Fetch main, stop the bot, fast-forward and restart in the existing tmux pane.
powershell -NoProfile -File scripts/deploy_vps.ps1 -Deploy

# Explicitly select the TWO recommended strategies with Terra 40% and live orders.
# Run this yourself only when ready to activate.
powershell -NoProfile -File scripts/deploy_vps.ps1 -Deploy -Recommended
```

Target: `sihan@91.98.132.183`, `/home/sihan/crypto_trading_bot`, session
`crypto_bot`. Uses the running bot's Python executable and existing shell
environment. Requires SSH key authentication and one pane running
`python run_bot.py` directly beneath its shell.

The deployment refuses tracked local edits, divergent history, changed
requirements, unexpected processes, or concurrent deployment. Untracked remote
files are preserved; Git will refuse an update that would overwrite them.
An unchanged Git commit and unchanged config selection do not trigger a restart.
Switching to the recommended config does restart, even on the same commit. This command does not
commit/push local work or install dependencies.

`-Recommended` starts `run_bot.py` with `BOT_CONFIG=config.recommended.yaml`.
The ordinary command preserves the currently selected config; it is NOT a
rollback command. The existing `config.yaml` and `.env` are not overwritten
by config selection. The new tracked recommended config arrives via Git.
Recommended setup: BTC band50 on 4h plus BNB RSI2 pullback on daily candles;
both use Terra 40%. Fixed buy lots remain 0.0005 BTC and 0.05 BNB. Existing
wallet holdings are not adopted or sold by the new strategy journals.

Offline verification: `python scripts/verify_release.py`.
Optional provider check: `python scripts/check_recommended.py --review`
(two synthetic OpenAI requests, no exchange account calls/orders/notifications).
See `research/FINAL_RECOMMENDATION.md` for results and limitations.

A successful restart means the process survived ten seconds; it does not prove
that API calls or orders work. Starting the existing bot triggers its normal
startup trading evaluation. If updating or restarting fails after shutdown,
inspect the session manually; there is no automatic rollback or retry:

```powershell
ssh -t sihan@91.98.132.183 tmux attach -t crypto_bot
```

Detach without stopping the bot using Ctrl-B, then D.
