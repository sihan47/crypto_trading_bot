# Terra risk-overlay integration

The user's risk-review framework is included as the system instruction on every actual decision-model request. The existing closed-bar/eligibility cadence is unchanged: a four-hour bar does not force an unnecessary GPT request, and the heartbeat remains independent.

The model emits exactly five fields: `red_team_counterargument`, `catalyst_and_flow_alignment`, `veto_triggered`, `size_multiplier`, `execution_note`. Local validation rejects extra/missing keys, non-finite or boolean sizes, empty explanations, and contradictory veto/size pairs. Invalid/incomplete model output follows the existing no-order failure path.

The execution system is spot, not perpetual, and the existing technical council has not become a statistically validated microstructure strategy. The prompt states both facts. It reviews an engine-proposed eligible direction rather than inventing one. Where the council baseline is HOLD but already exposes technically eligible directions, the engine deterministically proposes the eligible direction with the largest technical score; the model cannot choose a different side.

The exact five-field schema has no confidence field. An approval therefore becomes a binary support vote of 100 and a veto 0 for the existing weighted gate (configured 60% technical / 40% GPT); this number is explicitly not a probability. A veto overrides even a strong technical score. Approval still must pass the technical weighted gate. This differs from the former model-supplied graded confidence and should be understood before deployment.

Positive size multipliers reduce BUY quantity before exchange lot/notional rounding; below-minimum sizes fail validation instead of being rounded up. SELL remains a full strategy-owned spot exit when approved, avoiding unmanaged residual positions and accidental liquidation of unrelated balances. The requested macro veto is for increasing exposure; the prompt does not block a risk-reducing exit merely because a release is imminent.

## Available inputs and current limits

Existing funding snapshot, fear/greed, strategy cards and news summary remain in the payload. There is currently **no connected real-time anomaly-percentile feed or authoritative macro-calendar feed**. The payload says unavailable, not a fabricated quiet calendar. The policy includes FOMC/CPI/NFP/PCE and structural-shock veto logic, but those cannot be claimed as independently monitored or deterministically enforced scheduled-event gates without those inputs. A model has no implicit live browsing in this API call.

The output schema accepts 0–1.5 as requested. While live telemetry/calendar are unavailable, effective execution sizing is capped at 1.0; a model's unsupported 1.5 request is recorded but not executed. The original requested news/funding interaction is conditional on evidence; mildly bearish news is not automatically assumed priced in. A risk review is not evidence that the new research candidates have been deployed.

## Verification and deployment state

Offline tests verify the actual system message, strict response handling, veto precedence, directional restrictions, sizing caps, and runner quantity reduction before exchange rounding. No live API call, Telegram message, order, VPS restart or deployment is part of this change. The prompt becomes active on VPS only after the code is deployed/restarted through the existing workflow. No config or secret file change is required.
