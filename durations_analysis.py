# We'll implement a robust parser and plotter for the durations chart.
# It will look for the log at "results_comparison_90/execution_log.txt" first,
# and fall back to "execution_log.txt" if the first path is not found.

import os
from pathlib import Path
import re
from datetime import datetime
import locale
import matplotlib.pyplot as plt

# Ensure locale "C" so that English month/day abbreviations parse correctly,
# regardless of the user's OS locale (e.g., pt_BR).
try:
    locale.setlocale(locale.LC_TIME, "C")
except Exception:
    pass  # If setting locale fails, we'll still try to parse with default

# Resolve possible locations for the log file
base_dir = Path("/results_comparison_200")
preferred_path = base_dir /"execution_log.txt"
fallback_path = base_dir / "execution_log.txt"

if preferred_path.exists():
    log_path = preferred_path
elif fallback_path.exists():
    log_path = fallback_path
else:
    raise FileNotFoundError(
        "Não encontrei o arquivo de log em '/mnt/data/results_comparison_200/execution_log.txt' "
        "nem em '/mnt/data/execution_log.txt'."
    )

# Read the log content
with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# Regex to capture strategy blocks
block_pattern = re.compile(
    r"=== ESTRATÉGIA: (\w+) ===\s*"
    r"Início: (.*?)\n\s*"
    r"Status: SUCESSO\s*"
    r"Fim: (.*?)\n",
    re.MULTILINE
)

matches = block_pattern.findall(log_content)
if not matches:
    raise RuntimeError("Nenhum bloco de estratégia válido foi encontrado no log.")

# Parse the dates
date_format = "%a, %b %d, %Y %I:%M:%S %p"
durations = {}
for strategy, start_str, end_str in matches:
    start_clean = " ".join(start_str.strip().split())
    end_clean = " ".join(end_str.strip().split())
    start_dt = datetime.strptime(start_clean, date_format)
    end_dt = datetime.strptime(end_clean, date_format)
    durations[strategy.lower()] = (end_dt - start_dt).total_seconds()

# Sort by strategy name for a stable order
sorted_strategies = sorted(durations.keys())
seconds = [durations[s] for s in sorted_strategies]

# Create out directory
out_dir = base_dir / "comparison_graphics_90"
out_dir.mkdir(parents=True, exist_ok=True)

# Plot (using default Matplotlib settings; no custom colors/styles)
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_strategies, seconds)  # no color specified (tool guideline)
plt.xlabel("Estratégia de Agregação")
plt.ylabel("Duração (segundos)")
plt.title("Duração do Experimento por Estratégia")
plt.grid(True, axis="y", alpha=0.3)

# Add minute:second labels on top of bars
for bar in bars:
    height = bar.get_height()
    minutes = int(height // 60)
    sec = int(height % 60)
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{minutes}:{sec:02d}",
             ha="center", va="bottom", fontsize=10)

plt.tight_layout()
out_file_png = out_dir / "experiment_durations.png"
plt.savefig(out_file_png)
plt.close()

# Also save a PDF version if desired
out_file_pdf = out_dir / "experiment_durations.pdf"
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_strategies, seconds)
plt.xlabel("Estratégia de Agregação")
plt.ylabel("Duração (segundos)")
plt.title("Duração do Experimento por Estratégia")
plt.grid(True, axis="y", alpha=0.3)
for bar in bars:
    height = bar.get_height()
    minutes = int(height // 60)
    sec = int(height % 60)
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{minutes}:{sec:02d}",
             ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(out_file_pdf)
plt.close()

(str(out_file_png), str(out_file_pdf))
