# 肺結節分割專題快速完結版

## 一句話成果

本專題以 PVTFormer 為基礎，將原本的肝臟 CT 分割架構轉換至 LUNA16
肺結節任務，建立前、中、後三張相鄰切片的 2.5D 輸入流程，並比較
Voxel Attention、Attention Gate 與 Coordinate Attention 三種融合方法。

## 原專題主要結果

| 模型 | Recall | F1 / Dice | Precision | Jaccard / IoU | F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PVTFormer baseline | 0.7953 | 0.7273 | 0.8979 | 0.7070 | 0.7164 |
| Voxel Attention | 0.8130 | 0.7917 | 0.9509 | 0.7757 | 0.7917 |
| Attention Gate | 0.8168 | **0.7994** | **0.9555** | **0.7827** | **0.7983** |
| Coordinate Attention | **0.8208** | 0.7899 | 0.9384 | 0.7718 | 0.7892 |

Attention Gate 在 F1、Precision、IoU 與 F2 上表現最好，表示它對降低
錯誤分割並維持區域重疊較有幫助；Coordinate Attention 的 Recall 最高，
代表它較能保留可能的結節區域。對肺結節篩檢而言，背景切片同樣是實際
輸入的一部分，因此本專題保留包含空白背景切片的整體切片評估。

## 可直接放進備審的專題摘要

本研究以 PVTFormer 作為基礎模型，將原先應用於肝臟 CT 分割的架構
調整至 LUNA16 肺結節影像。為補足單張切片缺少上下文資訊的問題，建立
前、中、後三張相鄰切片的 2.5D 輸入流程，並分別實作 Voxel Attention、
Attention Gate 與 Coordinate Attention 進行比較。實驗結果中，Attention
Gate 取得最佳 F1 0.7994、Precision 0.9555 與 IoU 0.7827；Coordinate
Attention 則取得最高 Recall 0.8208。結果顯示，不同注意力機制會在降低
假陽性與保留疑似結節區域之間形成不同取捨。

## 面試時的30秒說法

「我的專題是把原本用於肝臟分割的 PVTFormer 遷移到 LUNA16 肺結節
任務。我負責資料處理、2.5D 相鄰切片流程、三種注意力模型的實作，以及
訓練和結果比較。最後 Attention Gate 的 F1 和 Precision 最好，Coordinate
Attention 的 Recall 最高。我從這個專題學到，不只要看單一準確率，還要
分析模型漏掉結節和誤判背景之間的取捨。」

## 被問到新舊數字差異時

「原專題採逐張切片計算後平均，因為真實使用情境也包含大量沒有結節的
背景切片；後來的重現實驗另外加入前景像素合併的 micro 指標，專門觀察
結節區域重疊。兩者評估目的不同，所以我把原專題數字作為主要成果，
新版結果作為補充分析，不直接混在同一張表比較。」

## 限制與未來工作

LUNA16 提供結節中心點與直徑，並非逐像素專家輪廓，因此本專題使用
幾何方式建立分割標籤。未來若延續研究，會改用具專家輪廓的資料、增加
病灶層級敏感度與每掃描假陽性數，並在相同資料切分和評估設定下重新
比較所有模型。

## 證據狀態

舊權重已從原 Linux 主機救回並成功載入。三種注意力權重可嚴格對應目前
模型；Baseline 使用 `PVTFormer-main/files0/checkpoint.pth` 作為可用候選。
原報告時期的完整資料切分 manifest 已不存在，因此原表應標示為「原專題
實驗結果」，不能宣稱是新版流程重新產生的數字。

## 簡易 Demo

專案已提供使用救回之原始 Attention Gate 權重的單病例展示：

```bash
./scripts/run_quick_legacy_demo.sh
```

它會自動選擇示範病例中標註面積最大的肺結節切片，輸出輸入 CT、參考
遮罩、機率熱圖、二值預測與疊圖。若要展示空白背景切片，可改用
`demo_infer.py --slice-index <編號>` 明確指定。此 Demo 使用現存的
`sphere-v1` 資料展示舊權重的實際推論能力，不宣稱重建已遺失的原報告
資料切分。

預設展示病例 `nodule_592` 的單切片結果為 Dice 0.7822、IoU 0.6423、
Recall 1.0000、Precision 0.6423。這組數字只用於說明該次 Demo，不代表
整個測試集的整體成績。
