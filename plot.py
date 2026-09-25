import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

csv_data = """
config,jsd_mean,mmd2,erp_amp_err,erp_lat_err_ms
  baseline,0.0208496165869291,0.014411257225346,0.0113163674250245,42.81662217336684
  postnet_only,0.0123844014742644,0.0134975483000374,0.0133437793701887,39.38147770100503
  dwt,0.0016758999900048,0.0015585162818466,0.0064294887706637,13.617894158291463
  dwt_stacking,0.0018647366696313,0.0007979413538942,0.01170589402318,37.05048680904522
  fm_lambda_0,0.0014371477600434,0.0003439361336264,0.008887137286365,14.84473146984924
  fm_lambda_20_postnet,0.0038243885719566,0.0035251720315798,0.009465486742556,19.1386620603015
  fm_lambda_50,0.0014564189496013,-5.476714709407027e-05,0.0057865716516971,14.108629082914575
  fm_lambda_0_postnet,0.0029568164263764,0.0037214398386005,0.0088381245732307,17.66645728643215
  fm_lambda_50_postnet,0.0015985002046363,0.0052967566420528,0.0099113313481211,20.242815640703505
  eeg_gan_vanilla_time,0.0428352172020822,0.0906480209117257,0.0741273462772369,116.54954459798992
  eeg_gan_vanilla_channels,0.0311544227006379,0.1827306100843727,0.0693387687206268,121.21152638190956
  eeg_gan_vanilla_full,0.026713648228906095,0.018987408945714046,0.02090410143136978,54.59426036432161
"""

df = pd.read_csv(StringIO(csv_data))
# Reverse to keep the table order from top to bottom in the horizontal bar chart
df = df.iloc[::-1].reset_index(drop=True)

sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(
    "Desempeño de Modelos EEG-GAN (Menor valor = Mejor rendimiento)",
    fontsize=22,
    fontweight="bold",
    y=0.98,
)

# JSD Mean
sns.barplot(data=df, x="jsd_mean", y="config", ax=axes[0, 0], palette="Blues_r")
axes[0, 0].set_title("Divergencia Jensen-Shannon (JSD Mean)", fontweight="bold")
axes[0, 0].set_xlabel("Valor (menor es mejor)")
axes[0, 0].set_ylabel("")

# MMD2
sns.barplot(data=df, x="mmd2", y="config", ax=axes[0, 1], palette="Greens_r")
axes[0, 1].set_title("Distancia Máxima de la Media (MMD²)", fontweight="bold")
axes[0, 1].set_xlabel("Valor (menor es mejor)")
axes[0, 1].set_ylabel("")

# ERP Amp Error
sns.barplot(data=df, x="erp_amp_err", y="config", ax=axes[1, 0], palette="Oranges_r")
axes[1, 0].set_title("Error de Amplitud ERP", fontweight="bold")
axes[1, 0].set_xlabel("Error")
axes[1, 0].set_ylabel("")

# ERP Latency Error
sns.barplot(data=df, x="erp_lat_err_ms", y="config", ax=axes[1, 1], palette="Reds_r")
axes[1, 1].set_title("Error de Latencia ERP (ms)", fontweight="bold")
axes[1, 1].set_xlabel("Milisegundos (ms)")
axes[1, 1].set_ylabel("")

plt.tight_layout(rect=[0, 0, 1, 0.95])
with open("./eeg_gan_metrics.png", "wb") as f:
    fig.canvas.print_png(f)
print("Plot generated successfully.")
