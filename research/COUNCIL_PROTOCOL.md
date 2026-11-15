# Council v1：評估前固定的規則

## 2026-11-15 後續刪除審查

已看過六策略結果後，固定只刪除 BNB Donchian，再做一次前後比較；不搜尋所有子集合或調整閾值來挑最高收益。BTC 保留六個，BNB 保留五個。刪除後保留所有存續專家的相對權重，統一正規化為 1，避免將整份突破組權重交給 Keltner。此為事後探索性審查，不是未見樣本測試，也不表示虧損達到統計顯著。

基準重現：`python research/evaluate_council.py`。刪除前後比較：`python research/evaluate_pruned_council.py`。兩者共用 runtime 計算；基準明確不傳 symbol 以保留原六策略，現行引擎傳入 symbol 套用刪除名單。

## 原始六策略規則

2026-11-15。已看過此前全區間結果；本輪也是探索性研究，不是未見樣本驗證。
不依這輪收益調整參數或挑最佳組合。兩幣都執行相同六策略，4h 收盤評估。

* 趨勢組：50 日 SMA ±1%、84 日動能 ±1%、65 日 SMA 標準差偏離 ±0.5。三者均分組權重。
* 突破組：前 20 日高點進、前 10 日低點出；日線 EMA20 +2 ATR10 進、跌回中線出。兩者均分組權重。
* 回撤組：日線 RSI2 下穿 15 且高於 SMA100；反彈上穿 30 或跌破 SMA100 提供退出意見。
* 日線只由完整六根已收盤 4h K 線組成；日線事件保留 24h。反彈退出是組合意見，並不聲稱現有持倉一定由回撤策略建立。
* 有限指數平滑避免不同歷史長度造成漂移：EMA20 使用最近 100 日的指數權重，RSI2 漲跌均值使用最近 30 日；ATR10 為日線真實波幅簡單平均。這是明確的改編，非原書精確重現。
* 行情以 20 日效率比 abs(close-close[120])/sum(abs(diff),120) 定義：≥.25 趨勢、≤.10 盤整，其餘混合。
* 趨勢／突破／回撤組權重：趨勢行情 .50/.30/.20；盤整 .30/.25/.45；混合 .40/.30/.30。
* BUY、SELL 的有效權重正規化成技術分數；HOLD 是沒有意見。有效覆蓋率至少 .20。分組限制相關指標重複投票，但不宣稱組間完全獨立。
* 持倉只能 HOLD/SELL，空倉只能 HOLD/BUY。Terra 最後選擇；HOLD 必定不交易；方向須有技術支持，且 .6×技術分數+.4×GPT 信心 ≥60(BUY)/55(SELL)。信心不是獲利機率。
* 技術代理回測不使用 GPT：用技術分數直接達 60/55 為進出場；不能將代理結果稱為 Terra 組合績效。
* 每個幣各自全資金、現貨多頭、收盤決策下一根開盤成交；15m 市值衡量；每邊費用 .1% + 滑價 .05%，另做雙倍成本、逐年及個別策略／相關性比較。
* 最近 601 根 4h 資料必須連續有效；不足或有缺口先不形成可交易證據。保留所有候選結果，不只報最佳。

來源：

* [StockCharts Keltner](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels)：EMA 與 ATR 通道及順勢解讀；本版使用上述有限平滑改編。
* [QuantConnect Alpha 架構](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/alpha/key-concepts)：各 Alpha 提供可識別的意見，再交由組合層整合。
* [論壇多策略如何整合](https://www.reddit.com/r/algotrading/comments/1ghawod/)：獨立下單、訊號一致性、整合交易的取捨；只作設計靈感，未將留言收益當證據。
* [NBER 加密資產動能](https://www.nber.org/papers/w24877)、[Turtle 通道](https://www.turtletrader.com/rules/)、[Alvarez RSI 回撤原規則](https://alvarezquanttrading.com/wp-content/uploads/2019/02/StrategyDescriptionResults.pdf)：相關研究與原始規則；本版是加密現貨改編。
