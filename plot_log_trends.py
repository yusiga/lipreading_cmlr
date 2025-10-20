import re
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
try:
    # Windows 常用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
except:
    # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 日志文件路径
log_file = "log/2025.10.17_15.44.57-1011-notsc.log"

# 正则表达式匹配 epoch、cer、loss
pattern = re.compile(
    r"epoch\s+(\d+)\s+valid ends:.*?\n.*?cer=([\d\.]+).*?loss=([\d\.]+)",
    re.DOTALL
)

epochs, cers, losses = [], [], []

# 读取日志内容
with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()

for match in pattern.finditer(log_text):
    epoch = int(match.group(1))
    cer = float(match.group(2))
    loss = float(match.group(3))
    epochs.append(epoch)
    cers.append(cer)
    losses.append(loss)

print(f"共解析出 {len(epochs)} 个 epoch")

# 使用现代配色
plt.style.use('seaborn-v0_8-muted')
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ---------- Loss 图 ----------
axes[0].plot(epochs, losses, color='#E67E22', marker='o', markersize=6, linewidth=2, label='Loss')
for i in range(0, len(epochs), 5):
    axes[0].text(epochs[i], losses[i] + 0.15, f"{losses[i]:.2f}", ha='center', fontsize=8, color='#E67E22')
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Loss", fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

# ---------- CER 图 ----------
axes[1].plot(epochs, cers, color='#27AE60', marker='o', markersize=6, linewidth=2, label='错误率')
for i in range(0, len(epochs), 5):
    axes[1].text(epochs[i], cers[i] + 0.01, f"{cers[i]:.3f}", ha='center', fontsize=8, color='#27AE60')
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("错误率", fontsize=12)
axes[1].set_title("错误率", fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
