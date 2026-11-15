# BTC 衍生品歷史資料已建立

2026-11-15。全部由 Binance 公開資料取得，沒有讀取 `.env`、使用帳戶金鑰、下單或操作 VPS。這是研究快照，不是常駐自動更新服務，也尚未接入正式 bot/GPT 決策。

## 檔案位置與內容

資料目錄：`data/research_structural/`。時間一律 UTC，快照主要涵蓋 2022-09-01 至 2026-11-13。

| 檔案 | 用途 |
| --- | --- |
| `btc_hourly_research.csv` | 可直接供研究讀取的聯合資料表：36,840 個小時；價格、OI、各類多空比、已結算 funding、Spot/Perp 及 Mark/Spot 基差 |
| `btc_metrics_5m.csv` | 441,947 筆清理後 metrics，保留原始 `create_time`、可用時間與來源日檔 |
| `fundingRate.csv` | 4,605 次實際 funding 結算；不是未來預測費率 |
| `klines.csv` / `markPriceKlines.csv` | 各 36,840 根永續成交價／標記價小時 OHLC |
| `metrics_archives/` | 1,535 個原始日檔 ZIP 及官方 SHA-256 CHECKSUM，全部驗證通過 |
| `metrics_download_manifest.json` | 每個下載檔的日期、URL、hash、列數；沒有缺失的整日檔案 |
| `rest_crosscheck/` | 抽查 2026-11-13 的五個 REST 端點原始回應 |
| `README.md` | 欄位定義及時間規則 |

現貨價格來源另保留在 `data/research_rotation/BTCUSDT.csv` 與 `data/research_15m/BTCUSDT.csv`。原始資料未上 GitHub；下載、建表、測試與稽核報告有版本管理。

## 讀取方式

```python
import pandas as pd
panel = pd.read_csv("data/research_structural/btc_hourly_research.csv", index_col=0)
panel.index = pd.to_datetime(panel.index, utc=True)
as_of = pd.Timestamp("2026-11-13 12:00", tz="UTC")
known = panel.loc[panel.index <= as_of]
# 再依假說需要的欄位檢查缺值；不得使用後面的資料回填。
```

索引是 **decision_time，並非 K 線開盤時間**。該列的價格 K 線已收盤；OI 與 funding 只向後匹配已到可用時間的紀錄。不要把整份歷史檔送給某個過去時間點的 GPT 決策，造成未來資料洩漏。

## 時間與欄位不能混用

* OI 數量 `sum_open_interest` 和美元價值 `sum_open_interest_value` 不同。後者會隨價格改變，不能直接當新增槓桿。
* `count_long_short_ratio` 是帳戶數多空比，不是市場淨多單 BTC 數量。
* `count_toptrader_long_short_ratio` 是大戶帳戶數比；`sum_toptrader_long_short_ratio` 是大戶部位比。
* `sum_taker_long_short_vol_ratio` 是 taker 買賣量比。比值不是可加總的成交量，不可直接平均後宣稱精確的總成交量失衡。
* `perp_spot_basis_bps` 是對齊小時收盤價計算的價差基點，不是可成交 bid/ask 套利價差。
* `metric_available_at = create_time +65 分鐘`：5 分鐘標籤不確定性加 60 分鐘發布緩衝。這是保守研究假設，不是已驗證的歷史發布日誌。
* funding 可用時間為實際結算時刻加一小時；只代表已知的過去費率，不表示下一次也會相同。
* metrics 超過 10 分鐘匹配容許窗口不延用；缺值保持 NaN。不把 OI 缺值補成零，也不把缺失多空比補成中性。

抽查日顯示 `日檔(T−5 分鐘)` 的 OI 數量及價值與 `REST(T)` 相符；多空比接近但不完全一致。taker 比的時間對應又不同。這只是一天的交叉核對，沒有宣稱驗證所有歷史時間標籤。詳見 [REST 比對](btc_metrics_rest_crosscheck.json)；研究沒有依回測收益去選時間位移。

## 品質稽核

36,840 個小時中，36,793 列具備核心 OI、一般帳戶多空比、funding、基差及可交易現貨資料，約 99.87%。這不代表所有大戶欄位也完整。

| 小時級欄位 | 缺值列數 |
| --- | ---: |
| OI 數量 | 45 |
| OI 美元價值 | 46 |
| 一般帳戶多空比 | 16 |
| 大戶帳戶多空比 | 2,532 |
| 大戶部位多空比 | 2,528 |
| Taker 買賣量比 | 12 |
| 已知 funding | 1 |
| 現貨／永續價差 | 1 |

原始資料有四處超過 5 分鐘的時間缺口，最大 630 分鐘；雖然每天的 ZIP 都存在，日內資料仍可能不完整。這次未發現互相衝突的重複時間戳；程式仍會隔離此類衝突，不採最後一筆覆蓋。零 OI、非有限數值與不合法負值會標記缺失。
完整逐年涵蓋率、無效值數及產物 SHA-256 在 [資料稽核](btc_derivatives_data_audit.json)。

## 使用新資料的第一個假說：淘汰

先發布 [四步研究備忘錄](OI_RESEARCH_MEMO.md)，再測「事件前多頭擁擠、價格及 OI 數量同時異常下降後的現貨回復」。只用三個入場特徵，沒有堆疊 funding、基差、大戶比來湊最佳組合。

與前輪相同：2023–2024 開發診斷、2025–2026/11/13 時間留出；歷史已曝光，不是純未見樣本。起始資金 10,000 USDT，計畫單筆風險 0.5%，不槓桿，現貨每邊費用 0.10%＋滑價 0.05%。只做現貨，因此交易 funding 為零；不是忽略永續 funding。事件後等待資料可用，再於下一根 15m 開盤成交；事件低點失效，最多 24h，含跳空、入場當根止損及成交量限制。

| 留出期結果 | 數值 |
| --- | ---: |
| 淨報酬 | −0.88% |
| 最大回撤，15m 市值 | −4.78% |
| 日報酬 Sharpe | −0.12 |
| Profit Factor | 0.92 |
| 完整交易 | 32 |
| 雙倍成本報酬 | −4.30% |
| 額外延遲資料 1h / 2h | −1.45% / −0.48% |
| ±20% 及共同擾動正報酬情境 | 2 / 19 |

全區間淨報酬 −11.91%。留出低波動背景 19 筆合計 −483.79 USDT，高波動背景 13 筆 +396.23；止損退出 22 筆損失 −1,066.63，十筆時間退出賺 +979.08。這說明該去槓桿代理仍會買到持續下跌，不能因某個事後分組獲利就直接加濾網宣稱有效。

**判定不通過，不加入正式 bot。** OI 下降不等於「被迫平倉已結束」，帳戶比也不能辨認主動買方是否承接。後續若測試 taker 流量／基差重新定價，應先另訂假說與否證條件，不能繼續調窗口救這次結果。完整成本、逐年、交易分組與敏感度見 [回測結果](structural_deleveraging_evaluation.json)。交易與淨值 CSV 位於 `output/structural_deleveraging/`。

## 重現

```powershell
python research/fetch_btc_metrics.py
python research/build_btc_derivatives_panel.py
python research/crosscheck_btc_metrics.py
python -m unittest discover -s tests -p test_derivatives_panel.py
python -m unittest discover -s tests -p test_deleveraging.py
python research/structural_deleveraging.py
```

funding／永續／現貨輸入沿用前輪已保存資料，相關抓取程式為 `fetch_structural_derivatives.py`、`fetch_rotation_data.py`、`prepare_ensemble_data.py`。REST 交叉核對的歷史窗口有限；已保存原始回應，未來應使用快取或另選當時最近的完整日。

## 來源

[Binance 公開日檔](https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/)、[官方資料倉庫](https://github.com/binance/binance-public-data)、[官方連接器市場資料介面](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)。官方連接器說明 OI／多空比 REST 歷史窗口約最近一個月；私有金鑰不會把它變成完整歷史。
[時間標籤問題回報](https://github.com/binance/binance-public-data/issues/491)是社群回報，非交易所完整確認；本輪另作一天的 REST／日檔數值核對，保留統一延遲與限制。
