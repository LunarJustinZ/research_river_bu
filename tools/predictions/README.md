# Prediction CSV tools / 预测CSV工具

Two small command-line tools for a **saved prediction CSV**. They do not take a model, and they do not train, run inference or extract features.
两个针对**已保存预测CSV**的小工具。它们不接收模型，也不执行训练、推理或特征提取。

| tool / 工具 | what it does / 作用 | needs / 依赖 |
|---|---|---|
| `check_prediction_manifest.py` | Checks that the CSV has the same images, dates and labels as a split file (for example `data/discharge/val.csv`). 核对预测文件与指定清单的图片、日期、标签是否一致。 | Python standard library / 仅标准库 |
| `evaluate_predictions.py` | Computes the scores with the original `day_scores()` of `baseline/train_linear.py` at commit `4f0ef92`. 用4f0ef92版本的原 `day_scores()` 计算成绩。 | numpy, pandas |

Copy the whole `tools/predictions/` folder, including `vendor/`. `evaluate_predictions.py` will not run without `vendor/train_linear_4f0ef92.py`.
请复制整个 `tools/predictions/` 目录（包括 `vendor/`）。只拷贝 `evaluate_predictions.py` 一个文件无法运行。

## Input / 输入

Required columns / 必需列：`image_id`, `date` (YYYY-MM-DD), `discharge_lps`, `pred_discharge_lps`.
- Both discharge columns are in **L/s**. 两个流量列单位都是 L/s。
- Evaluate **one site per run**. If a `site` column is present it must hold a single non-empty value. Without a `site` column, you must make sure all rows come from one site; the tool does not guess the site from `image_id`.
  一次只评价**一个站点**。有 `site` 列时，它必须非空且只有一个值。没有 `site` 列时，由使用者保证输入只来自一个站点；工具不会根据 `image_id` 推测站点。
- `baseline/results/predictions_<split>.csv` written by `baseline/train_linear.py` already has these columns.
  `baseline/train_linear.py` 写出的 `baseline/results/predictions_<split>.csv` 已经包含这些列。

## Use / 使用（run from the repository root / 在仓库根目录运行）

On macOS/Linux use `python3` if `python` is not Python 3. **`--out` must be a directory that does not exist yet**; the tools never overwrite.
macOS/Linux 上如果 `python` 不是 Python 3，请改用 `python3`。**`--out` 必须是尚不存在的目录**，工具不会覆盖已有目录。

On Windows without long-path support, keep the full output file paths under 260 characters. A longer path makes writing the output fail.
Windows 未开启长路径支持时，输出文件的完整路径请保持在260个字符以内，否则写出结果会失败。

**Step 1: check against the split file first / 第一步：先对照清单**

```
python tools/predictions/check_prediction_manifest.py --predictions baseline/results/predictions_val.csv --reference data/discharge/val.csv --expected-reference-blob eb5951c6cb52c7582e4209b07180e896e5a89edc --out runs/check_val
```

Expected output / 预期输出：`{"status": "required_match", "differences_rows": 0}`, exit code / 退出码 0.

- `--expected-reference-blob` is optional. It makes the tool refuse a `val.csv` that is not the 4f0ef92 version (git blob `eb5951c6…`).
  `--expected-reference-blob` 可省略。加上后，如果 `val.csv` 不是 4f0ef92 版本（git blob `eb5951c6…`），工具会拒绝运行。

**Step 2: compute the scores / 第二步：计算成绩**

```
python tools/predictions/evaluate_predictions.py --input baseline/results/predictions_val.csv --out runs/eval_val
```

This prints the metrics. For the committed validation predictions: `days` 364, `within_x2` 0.9752747252747253 (355 of 364 days), `r2_log` 0.9470870…, `rmse_log` 0.2977664…, `spearman` 0.9746276…
运行后打印各项指标。对仓库中已提交的验证集预测，结果为：`days` 364，`within_x2` 0.9752747252747253（364天中355天达标），`r2_log` 0.9470870…，`rmse_log` 0.2977664…，`spearman` 0.9746276…

`runs/` is only an example output location; do not commit it. `runs/` 只是示例输出位置，请不要提交。

## Outputs / 输出

`check_prediction_manifest.py` → `--out/`
- `summary.json`: status, id counts, per-field results and empty-value counts. 状态、ID计数、逐字段结果及空值计数。
- `differences.csv`: **one row per difference; look here when the status is not `required_match`.** Issue types are `missing_in_predictions`, `extra_in_predictions`, `field_mismatch` and `empty_in_predictions_when_reference_present`.
  **每处差异一行；状态不是 `required_match` 时请看这里。** 问题类型包括 `missing_in_predictions`、`extra_in_predictions`、`field_mismatch`、`empty_in_predictions_when_reference_present`。
- A field that is empty on both sides (for example `fewshot` in `val.csv`) is counted in both `empty_in_*` counts and is not a difference.
  两侧都为空的字段（例如 `val.csv` 的 `fewshot`）会分别计入两边的空值数，但不算差异。

`evaluate_predictions.py` → `--out/`
- `metrics.json`: `days`, `within_x2`, `r2_log`, `rmse_log`, `spearman` as returned by `day_scores()`. A non-finite value (for example R² with only one day) is written as `null`, never as 0.
  `day_scores()` 返回的 `days`、`within_x2`、`r2_log`、`rmse_log`、`spearman`。非有限值（例如只有1天时的R²）写为 `null`，不会写成0。
- `daily_scores.csv`: one row per date with `y` and `p` (daily means on the log1p scale) and `hit` (|p − y| ≤ ln 2).
  每天一行：`y`、`p`（log1p刻度下的日均值）和 `hit`（|p−y| ≤ ln 2）。
- `evaluation_record.json`: input hash, row and date counts, site check, function source, unit conversion and runtime.
  输入哈希、行数、日期数、站点检查、函数来源、单位转换和运行环境。

## Exit codes / 退出码

| code | `check_prediction_manifest.py` | `evaluate_predictions.py` |
|---|---|---|
| 0 | `required_match` | scores written / 已写出成绩 |
| 1 | valid input, differences listed in `differences.csv` / 输入有效但有差异，见 `differences.csv` | — |
| 2 | invalid input, reference blob mismatch, or `--out` already exists / 输入无效、参照文件blob不符，或输出目录已存在 | invalid input or `--out` already exists (no output written) / 输入无效或输出目录已存在（不写出任何结果） |

## Small example (constructed data, not research data) / 小例子（构造数据，不是研究数据）

`examples/` holds hand-made rows with `demo_*` ids.
`examples/` 中是手写的数据行，ID 为 `demo_*`。

```
python tools/predictions/check_prediction_manifest.py --predictions tools/predictions/examples/example_predictions_matching.csv --reference tools/predictions/examples/example_reference.csv --out runs/example_check
python tools/predictions/evaluate_predictions.py --input tools/predictions/examples/example_predictions_matching.csv --out runs/example_eval
python tools/predictions/check_prediction_manifest.py --predictions tools/predictions/examples/example_predictions.csv --reference tools/predictions/examples/example_reference.csv --out runs/example_mismatch
```

Expected results / 预期结果：
1. Exit 0 with `required_match`. 退出码0，`required_match`。
2. Exit 0 with `days` 2 and `within_x2` 1.0. 退出码0，`days` 2，`within_x2` 1.0。
3. Exit 1 with `partial_or_mismatch`. `runs/example_mismatch/differences.csv` lists `demo_3` as `missing_in_predictions` and `demo_4` as `extra_in_predictions`. A label written as `10.00` instead of `10.0`, or rows in a different order, are not differences.
   退出码1，`partial_or_mismatch`。`runs/example_mismatch/differences.csv` 列出 `demo_3`（`missing_in_predictions`）和 `demo_4`（`extra_in_predictions`）。标签写成 `10.00` 而不是 `10.0`、行顺序不同，都不算差异。

## Scoring rule is fixed to 4f0ef92 / 评分规则固定为4f0ef92版本

- `evaluate_predictions.py` does not reimplement the metric. It reads `day_scores()` verbatim from `vendor/train_linear_4f0ef92.py`, an unmodified copy of `baseline/train_linear.py` at commit `4f0ef92506ba00ad6eeda7ab27be7cdf7fc72009` (git blob `2a461fd31c7d941bb7ab882a18db2f5b13a324bb`).
  `evaluate_predictions.py` 没有重新实现评分。它从 `vendor/train_linear_4f0ef92.py` 中原样读取 `day_scores()`；该文件是提交 `4f0ef92506ba00ad6eeda7ab27be7cdf7fc72009` 中 `baseline/train_linear.py` 的未修改副本（git blob `2a461fd31c7d941bb7ab882a18db2f5b13a324bb`）。
- Before use, the tool checks the file's git blob and then runs only that one function. torch and matplotlib are not imported, and training is not run.
  使用前，工具会先核对该文件的 git blob，然后只执行这一个函数，不导入 torch、matplotlib，也不运行训练。
- If `baseline/train_linear.py` changes later, this tool keeps the 4f0ef92 rule and does not follow automatically. Updating the rule means replacing the vendor file and its hash on purpose.
  以后 `baseline/train_linear.py` 改变时，本工具仍使用4f0ef92的规则，不会自动跟随。如需更新规则，必须有意替换 vendor 文件及其哈希。

## Tests / 测试（constructed program tests; no images, no models / 构造的程序测试，不用图片和模型）

```
python tools/predictions/tests/run_manifest_tests.py runs/test_manifest
python tools/predictions/tests/run_evaluate_checks.py baseline/results/predictions_val.csv tools/predictions/tests/metrics_reference_val_4f0ef92.json runs/test_evaluate
```

- The first runs 27 constructed cases. 第一条运行27项构造测试。
- The second runs 17 constructed cases, plus an acceptance run on the committed validation predictions: 1810 rows, 364 days and 355 hit days, compared with the validation metrics of `baseline/results/metrics.json`, which are copied into `tests/metrics_reference_val_4f0ef92.json`.
  第二条运行17项构造测试，并对已提交的验证集预测做一次验收：1810行、364天、355个达标日，并与 `baseline/results/metrics.json` 中的验证集指标对照（这些指标已复制到 `tests/metrics_reference_val_4f0ef92.json`）。
- Both print JSON and exit 0 when everything passes. 全部通过时两条都打印JSON并以退出码0结束。

## What a match does not show / 一致不代表什么

A `required_match` and a set of scores show only that the CSV agrees with the split file and was scored with the 4f0ef92 rule. They do **not** show that the discharge labels are true, or that the model actually read the matching images. The `sha256` column is compared only as declared text; image bytes are not read.
`required_match` 和计算出的成绩只说明：预测文件与清单一致，并按4f0ef92规则计分。它们**不能**证明流量标签是真值，也不能证明模型确实读取了对应的图片。`sha256` 列只按声明的文本比较，不读取图片字节。
