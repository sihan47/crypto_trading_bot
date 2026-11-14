#!/usr/bin/env bash
# Sent over SSH by deploy_vps.ps1. Default mode is read-only.
set -euo pipefail
mode=${1:---check}
[[ "$mode" == --check || "$mode" == --deploy ]] || exit 2
cd /home/sihan/crypto_trading_bot
exec 9>/tmp/crypto_bot-deploy.lock
flock -n 9 || { echo 'Another deployment is running.'; exit 1; }
[[ $(git branch --show-current) == main ]] || { echo 'Expected main branch.'; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked remote files have changes; stopping.'; exit 1; }
mapfile -t panes < <(tmux list-panes -s -t crypto_bot -F '#{pane_id}')
[[ ${#panes[@]} == 1 ]] || { echo 'Expected exactly one pane in crypto_bot.'; exit 1; }
pane=${panes[0]}
shell_pid=$(tmux display-message -p -t "$pane" '#{pane_pid}')
[[ $(tmux display-message -p -t "$pane" '#{pane_current_path}') == "$PWD" ]] || exit 1
mapfile -t children < <(pgrep -P "$shell_pid" || true)
[[ ${#children[@]} == 1 ]] || { echo 'Expected one running bot directly under the tmux shell.'; exit 1; }
bot_pid=${children[0]}
mapfile -d '' -t argv < /proc/"$bot_pid"/cmdline
[[ ${#argv[@]} == 2 && ${argv[1]} == run_bot.py ]] || { echo 'Unexpected running command; stopping.'; exit 1; }
python_bin=$(readlink -f /proc/"$bot_pid"/exe)
[[ -x "$python_bin" ]] || exit 1
echo "Repo: $PWD | branch: main | commit: $(git rev-parse --short HEAD)"
echo "tmux: $pane | bot PID: $bot_pid | Python: $python_bin"
if [[ "$mode" == --check ]]; then
    echo 'Check passed. No bot restart or Git update performed.'
    exit 0
fi
git fetch origin main
git merge-base --is-ancestor HEAD origin/main || { echo 'Remote checkout cannot fast-forward; stopping.'; exit 1; }
if [[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]]; then
    echo 'Already up to date; leaving bot running.'
    exit 0
fi
# Avoid accidentally deploying incompatible dependencies without installing them.
if ! git diff --quiet HEAD origin/main -- requirements.txt requirements-dev.txt; then
    echo 'Dependencies changed. Update the environment explicitly before deployment.'
    exit 1
fi
echo 'Stopping bot with Ctrl-C...'
tmux send-keys -t "$pane" C-c
for ((i=0; i<30; i++)); do
    kill -0 "$bot_pid" 2>/dev/null || break
    sleep 1
done
if kill -0 "$bot_pid" 2>/dev/null; then
    echo 'Bot did not stop. No update or second process started.'
    exit 1
fi
trap 'echo "Deployment failed after stopping bot. Inspect with: ssh -t sihan@91.98.132.183 tmux attach -t crypto_bot"' ERR
git merge --ff-only origin/main
printf -v command 'cd %q && %q run_bot.py' "$PWD" "$python_bin"
tmux send-keys -t "$pane" -l "$command"
tmux send-keys -t "$pane" Enter
sleep 10
new_pid=$(pgrep -P "$shell_pid" -f 'run_bot.py' || true)
[[ -n "$new_pid" ]] || { echo 'Bot exited during startup.'; false; }
echo "Deployed $(git rev-parse --short HEAD); bot PID: $new_pid"
echo 'Process survived 10 seconds. This does not verify exchange connectivity or trading health.'
echo 'Inspect logs or attach: ssh -t sihan@91.98.132.183 tmux attach -t crypto_bot'
