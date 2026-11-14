# VPS deployment

Run from the repository root in Windows PowerShell after committing and pushing
the intended changes to GitHub `main`:

```powershell
# Check connection, checkout, tmux pane and running Python process only.
powershell -NoProfile -File scripts/deploy_vps.ps1

# Fetch main, stop the bot, fast-forward and restart in the existing tmux pane.
powershell -NoProfile -File scripts/deploy_vps.ps1 -Deploy
```

Target: `sihan@91.98.132.183`, `/home/sihan/crypto_trading_bot`, session
`crypto_bot`. Uses the running bot's Python executable and existing shell
environment. Requires SSH key authentication and one pane running
`python run_bot.py` directly beneath its shell.

The deployment refuses tracked local edits, divergent history, changed
requirements, unexpected processes, or concurrent deployment. Untracked remote
files are preserved; Git will refuse an update that would overwrite them.
An unchanged Git commit does not trigger a restart. This command does not
commit/push local work or install dependencies.

A successful restart means the process survived ten seconds; it does not prove
that API calls or orders work. Starting the existing bot triggers its normal
startup trading evaluation. If updating or restarting fails after shutdown,
inspect the session manually; there is no automatic rollback or retry:

```powershell
ssh -t sihan@91.98.132.183 tmux attach -t crypto_bot
```

Detach without stopping the bot using Ctrl-B, then D.
