# tests

## `test_download_replace.py`: re-downloading over a corrupt image / 重新下载覆盖损坏图片

**Problem / 问题**

`download.py` re-downloads an image whose file is present but has the wrong sha256. It used to save the verified download with `tmp.rename(dest)`. On Windows, `rename` refuses to overwrite an existing file (`FileExistsError`, WinError 183), so the download was reported as `failed`. The corrupt file stayed in place and the correct bytes were left behind in `<name>.part`, and running the command again failed the same way.

`download.py` 发现已有图片的 sha256 不对时会重新下载。原来用 `tmp.rename(dest)` 保存已通过校验的新内容；但在 Windows 上，`rename` 不能覆盖已存在的文件（`FileExistsError`，WinError 183），于是下载被报告为 `failed`：损坏的文件留在原位，正确内容只留在 `<name>.part` 里，再运行一次也会以同样方式失败。

**Fix / 修复**

`tmp.replace(dest)` overwrites the old file on every OS. The order is unchanged: the hash is checked first, the `.part` file is written next, and only then is the target replaced. A download with the wrong hash still never touches the target.

改为 `tmp.replace(dest)`，在所有系统上都会覆盖旧文件。顺序不变：先校验哈希，再写 `.part`，最后替换目标；哈希不对的下载仍然不会碰到目标文件。

**Nothing changes for users / 使用方式不变**

Commands, defaults, retries and hash checks are the same as before:
命令、默认值、重试和哈希校验都和以前一样：

```
python download.py --split val --limit 5
```

**Run the test / 运行测试**

Standard library only. No network is used and no image is downloaded:
只需标准库，不联网，也不下载图片：

```
python tests/test_download_replace.py -v
```

The test covers four cases / 测试覆盖四种情况：
- (a) no file yet → saved / 没有文件 → 保存；
- (b) correct file → skipped, no network call / 已有正确文件 → 跳过，不联网；
- (c) wrong file → replaced, then skipped on the next call / 已有错误文件 → 替换，下次调用时跳过；
- (d) download with the wrong hash → rejected, old file kept / 下载内容哈希不对 → 拒绝，保留旧文件。

With the previous `rename`, case (c) fails on Windows (checked on Windows 11 with Python 3.12). On macOS/Linux, Python documents that `rename` already overwrites an existing file; this was not tested there.
在 Windows 上，换成原来的 `rename` 时情况 (c) 会失败（已在 Windows 11、Python 3.12 上验证）。在 macOS/Linux 上，Python 文档说明 `rename` 本来就会覆盖已有文件；这一点没有在这两个系统上测试。
