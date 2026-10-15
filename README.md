# VoiceTool · 配音台

文字 + 一个演员的参考音 → mp3。独立工具，不依赖 Bookreel 的控制台；
引擎是 CosyVoice（零样本：没有内置音色表，**参考音就是声音**）。

在 `G:\work\VoiceTool` 里开一个终端（cmd 或 PowerShell）：

```
.\say --list                                    列出演员（voices/ 里的成对文件）
.\say -v storyteller "崇祯六年秋，苏州城厢，日头偏了。"
.\say -v 2 -f chapter01.txt -o ch01.mp3         按编号选演员，文字来自文件
.\say "天色渐暗。"                               不给 -v：从编号列表里选
.\say --audition "同一句试听的话。"               每个演员各出一个文件，用来挑人
.\say --web                                     开网页表单 http://127.0.0.1:9110
```

写 `.\say`，不是 `say`：这台机器的 cmd 不在当前目录找可执行文件
（`'say.bat' is not recognized` 就是这个）。从别的目录可以用全路径
`G:\work\VoiceTool\say.bat ...`。

第一次跑会自动执行 `setup.ps1`（也可以手动跑）。之后每次只是启动。
多段文字（带空行的）用 `-f 文件`：cmd 会把引号里的换行吃掉。

## 文件

| | |
|---|---|
| `say.bat` / `say.py` | 工具本体。`say.py` 可以在 Linux/macOS 用 `venv/bin/python say.py` 跑 |
| `setup.ps1` | 装环境：Python 3.10 venv、CUDA torch、CosyVoice 代码、依赖、权重。可重复跑，每步先检查 |
| `voices/` | 演员：`名字.wav`（3–10 秒干净人声）+ `名字.txt`（逐字转写）。见 `voices/README.txt` |
| `models/cosyvoice2/` | 权重。另有 `cosyvoice3` 时优先用它（`setup.ps1 -Model v3` 下载） |
| `third_party/CosyVoice/` | 推理代码（不在 PyPI 上） |
| `venv/` | Python 环境 |
| `output/` | 出来的 mp3；`--web` 的试听缓存在 `output/_audition/` |

## 加演员：把录音变成一对文件

```
.\say --add 录音.mp3 --name 乔峰 --work 天龙八部            整段用，whisper 转写
.\say --add 整集.mp4 --start 28.7 --end 34.6 --name 段誉    从长文件里截一段
.\say --add 录音.wav --name 旁白 --text "这段话的逐字转写。"   自己给转写（最准）
```

- 任何 ffmpeg 读得了的格式都行（mp3 / m4a / wav / mp4 …），出来的是 24 kHz
  单声道 wav，做了响度归一和淡入淡出（`--raw` 不归一）。
- 3–10 秒最好，超过 30 秒拒收：用 `--start/--end` 切。
- 不给 `--text` 就用 whisper 转写（默认 `small` 模型，`--whisper medium` 更准，
  首次会下载）。**转写一定要对着音频核一遍**再用：转写错一个字，声音就错一点。
  粤语片段 whisper 常写成普通话的字，要手改。
- 网页表单不用重开：演员列表每次刷新都重新读 `voices/`。
- 网页表单下面也有「加演员」：选文件、填名字/作品/起止秒数，上传后把转写
  核对一遍再「保存转写」，然后「试听这个演员」。「删演员」把当前选中的那
  个移到 `voices/_trash/`，不真删。
- 命令行对应：`.\say --set-text 乔峰 --text "改好的转写"`，`.\say --remove 乔峰`。

## 选项

| | |
|---|---|
| `-v 名字/编号` | 演员。省略则交互选 |
| `-f 文件` | 文字来自 utf-8 文件；`-f -` 读标准输入 |
| `-o 文件` | 输出路径。默认 `output/<演员>-<时间>.mp3` |
| `--format wav` | 直接出 24 kHz wav，不经 ffmpeg |
| `--speed 0.95` | 0.5–2.0 |
| `--instruct "请用低沉、克制、缓慢的语气讲述"` | 念法。**CosyVoice3 听得懂；CosyVoice2 会把这句话念出来**，v2 上别用 |
| `--device cpu` | 不用 GPU |
| `--model 目录名` | 指定 `models/` 里哪套权重 |
| `--server http://主机:9100` | 不加载模型，改用一个在跑的 Bookreel tts 服务 |

## 长文

超过一口气的文字按句切成 ≤180 字一段，逐段合成后拼起来，段间 0.18 秒停顿，
**空行**是 0.55 秒的段落停顿。单段太长 CosyVoice 会漂，所以切是必须的。
数字是它的弱项（三钱念成"30"过），写汉字。

## 本机实测（2026-10-15，RTX 5090，CosyVoice2）

| | |
|---|---|
| 模型加载 | 8–23 秒（首次慢，之后文件在缓存里） |
| 34 字 | 8.9 秒音频，合成 4 秒 |

第一次合成时 wetext 会从 modelscope 拉一次文本正则化的 fst（~1 MB，
缓存在 `~/.cache/modelscope`），之后离线。

## 为什么是这些依赖

- torch 走 PyTorch 自己的 cu128 索引：PyPI 上的 Windows 包是纯 CPU 版，
  而 RTX 50 系列需要 cu128 以上。
- torchaudio 2.9+ 自己不解码文件了（要 torchcodec，Windows 上装不干净）。
  `say.py` 把 `torchaudio.load` 换成 soundfile，CosyVoice 读参考音就从这里过。
- `requirements.txt` 不是上游那份：上游钉死 torch 2.3.1，还带训练和 demo
  的一堆包。文件里写了每一项为什么在。
- Python 必须 3.10：CosyVoice 的几个依赖在 3.13+ 没有 wheel。
