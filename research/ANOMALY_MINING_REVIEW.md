# Phase 1–2：四維資料與統計異常探索

本輪只做資料與 EDA。沒有策略進出場、部位、交易成本後 PnL、GPT 決策或實盤修改。Phase 3 尚未開始，等待共同審閱結果。

## Phase 1：已完成的資料集

- 標的：Binance USD-M BTCUSDT perpetual；統一 UTC 1h 格線，索引是資料決策時間。
- 期間：2022-09-01 01:00 至 2026-11-14 00:00，共 36,840 小時，最後一根代表 11/13 已收盤時段。
- 新下載並核對官方 SHA-256：63 份 K 線 ZIP（50 月檔、13 日檔）。保留原始 ZIP 與 CHECKSUM。
- 重新驗證既有 1,535 份官方每日 metrics 歸檔；沿用先前 Binance 公開 fundingRate REST 取得的實際資金費率結算歷史。
- 四類資料全部齊全共 34,277 小時。包含 top account 的完整率：2023 年 99.95%、2024 年 99.54%、2025 年 99.92%、2026 年 99.99%。2022 年頂級交易者欄位缺失較多，沒有回填或填零。

| 維度 | 欄位／處理 | 缺失小時 |
| --- | --- | ---: |
| 價量與主動買量 | Futures OHLC、base/quote volume、trades、taker buy base/quote volume | 1 |
| OI | `sum_open_interest`，合約基礎數量而非價格變動影響的 USDT 名目金額 | 45 |
| 資金費率 | 實際結算值，延後一小時才視為可得；最多沿用 12 小時 | 1 |
| Top 持倉多空比 | `sum_toptrader_long_short_ratio` | 2,528 |
| Top 帳戶多空比 | `count_toptrader_long_short_ratio` | 2,532 |
| Global 帳戶多空比 | `count_long_short_ratio` | 16 |

Top 持倉比與帳戶比不能混用；Global 包含各類帳戶，不是已分離的散戶族群。四種經濟維度也不保證統計上互相正交。完整來源、檔案雜湊與缺失數見 `orthogonal_data_audit.json`。

主動成交量不平衡：`I = (2 × taker_buy_base_asset_volume − volume) / volume`。這是成交方向代理，不是包含掛單新增／撤單的 order-book OFI。metrics 採原標籤 +65 分鐘的保守可得時間，非已驗證的歷史發布紀錄。缺失與陳舊資料保持 NaN。

來源：[Binance 官方公開資料格式](https://github.com/binance/binance-public-data)、[官方 Futures connector 的歷史窗口說明](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)。使用公開資料不需要 `.env` 或交易 API 金鑰。

## Phase 2：固定研究設計

程式：`mine_anomalies.py`，無交易規則。先按時間切出前 70%：2022-09-01～2025-08-10 12:00，共 25,788 小時。最後 30%（2025-08-10 13:00 起的 11,052 小時）不進入本輪特徵計算、分位數、標籤或統計檢定。這段資料曾用於先前研究，因此不能稱為整個專案從未見過的樣本外。

牛／熊定義是當時價格相對 90 天前的高／低，沒有使用未來峰谷。可分類牛市 17,022 小時、熊市 6,604 小時；前 90 天缺乏歷史，不分類。這只是可重現的趨勢分層，不等於所有經濟意義上的牛熊市場。

`z(x)` 使用此前 30 天均值與標準差，排除當前觀測。六項特徵在首次執行前固定：

| 特徵 | 定義 |
| --- | --- |
| Position/account divergence | `z(log(top_position_ratio) − log(global_account_ratio))` |
| Account/account divergence | `z(log(top_account_ratio) − log(global_account_ratio))`，相同口徑的對照 |
| OI-weighted funding | `z(funding) × exp(clip(log(OI) − prior_30d_mean(log(OI)), −3, 3))`；clip 僅防極端權重 |
| Divergence × OI change | 第一項乘上 `z(24h log OI change)` |
| Compression × flow acceleration | `Δ² I / sqrt(mean_24h(GK variance) / mean_168h(GK variance))` |
| Signed flow, low price impact | `I / (1 + abs(hourly return / prior_168h_return_std))`；不是直接觀察到買方補單 |

每項比較其此前 90 天的第 10／90 百分位尾端（至少 30 天有效觀測），分別對照同牛熊狀態下其餘時點。30／90 天、1 日／1 週是固定日曆尺度，沒有依結果搜尋最佳長度。其尺度敏感度尚未做；本輪不是穩健 Alpha 認證。

4／12／24h 的四種目標：未來累積 log return、`log(future_RMS_vol / past_168h_RMS_vol)`、報酬大於 `+2 × past_hourly_RMS × sqrt(h)` 的機率、小於負門檻的機率。缺一根未來報酬就不計該標籤，最後 h 小時也不跨入保留區間。

檢定量是「條件平均 − 同狀態其餘樣本平均」，避免把牛市本身的正報酬當成特徵效果。使用 168 小時 Bartlett Newey–West HAC 標準誤，保留實際小時格線處理重疊報酬及序列依賴；p 值是漸近常態近似。共 288 個可估計檢定，做 BH 與對檢定相依性更保守的 BY 校正。

原始門檻：牛熊同方向，兩邊各自 `abs(t)>2.5 或 p<0.01`。校正後資格：兩邊 BY q<0.01、各至少 100 個條件與對照小時、至少 30 個按預測期限錯開的事件。錯開事件數不是獨立性證明。偏態、超額峰度、5/95 分位數都有輸出，但未另外檢定其顯著性，不用描述性峰度冒充尾端優勢證據。

## 排名結果與解讀

144 組跨牛熊比較中，21 組達原始門檻、13 組通過原始 EDA 的 BY 門檻。後者為 12 組波動關聯與 1 組下尾機率關聯，集中於 **兩個特徵家族**。平均未來報酬沒有任何比較同時通過牛熊原始門檻。

下面列出三個代表性比較，而非三個獨立 Alpha。完整排序見 `output/anomalies/ranked_anomalies.csv`，全部檢定與偏態／峰度見 `all_tests.csv`。

| 比較，均為 4h | 牛市差異 | 熊市差異 | 牛／熊 HAC t | 較差一側 BY q |
| --- | ---: | ---: | --- | ---: |
| flow acceleration／compression 高尾 → log 波動比 | −0.2674 | −0.3116 | −13.92／−8.97 | 5.83e−17 |
| low-impact signed flow 低尾 → log 波動比 | −0.2758 | −0.2961 | −15.05／−8.26 | 2.21e−14 |
| low-impact signed flow 低尾 → 大幅下跌機率 | −1.536 百分點 | −2.048 百分點 | −4.24／−4.31 | 0.00127 |

前兩項表示後續波動比同狀態的其他時點低，沒有指示價格上漲。第三項表示按事前波動縮放的大跌事件較少，也不等於淨報酬為正。正負 flow 尾端皆顯示波動下降，更像活動程度／風險狀態訊息，而非方向性資訊優勢。

## 混淆因素診斷

看到上述結果後，另做明確標示的事後診斷 `check_anomaly_controls.py`。在相同探索資料，控制過去波動、當前絕對報酬、成交量、過去 24h 報酬、UTC 時段正餘弦及週末，再估計事件係數；仍採 168h HAC。把 6 個新增檢定加入原家族，共 294 個檢定計算 BY。這不是獨立驗證，控制變數也不能排除所有非線性混淆。

| 調整後的同三項比較 | 牛／熊係數 | 牛／熊 HAC t | 牛／熊 BY q |
| --- | --- | --- | --- |
| acceleration／compression → log 波動比 | −0.0769／−0.0795 | −4.40／−2.93 | 0.00067／0.10783 |
| low-impact flow → log 波動比 | −0.0675／−0.0752 | −4.14／−2.64 | 0.00195／0.23656 |
| low-impact flow → 大跌機率 | −0.325／−0.982 百分點 | −0.95／−1.97 | 1.0／1.0 |

效果明顯縮小；三項都未能在控制後同時通過牛熊 BY 門檻。結果支持「表面關聯部分由活躍度／波動狀態解釋」，不能證明完全沒有訊息。沒有為了得到三個候選而調低門檻、換分位數或打開保留區間。

## 交付與限制

- `data/research_structural/btc_orthogonal_1h.csv`：四維合併資料，與原始資料一樣留在本機，不提交 Git。
- `orthogonal_data_audit.json`：來源、完整性、缺值與 SHA-256。
- `anomaly_findings.json`：原始探索結果及排名摘要。
- `anomaly_control_diagnostics.json`：事後混淆控制結果。
- `output/anomalies/`：所有檢定、完整排序、探索特徵、Spearman 相關矩陣。
- 執行順序：`python research/ingest_orthogonal.py` → `python research/mine_anomalies.py` → `python research/check_anomaly_controls.py`。前者使用本專案已稽核的 metrics/funding/panel 快取；若尚未建立，先執行既有資料擷取與 panel 建置流程。

HAC p 值依賴大樣本近似；BY 也不能修正不準確的單項 p 值、資料品質或研究者選擇偏誤。成交量代理沒有 order-book 深度資訊。顯著統計關聯未扣交易成本，也不是交易績效；本輪刻意不計算虛構 Sharpe。

目前尚未找到三個獨立、控制混淆後仍穩健的報酬／尾端異常。停在 Phase 2 供審閱，不開始策略錦標賽。
