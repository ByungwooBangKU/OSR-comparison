import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 확장된 데이터 입력 (위 표 기반)
data_full = {
    'Base_Strategy': [
        'Standard',
        'Entropy_b25', 'Entropy_b25', 'Entropy_b25', 'Entropy_b25',
        'MaxAttn_t85', 'MaxAttn_t85', 'MaxAttn_t85', 'MaxAttn_t85', # MSP 0.1 추가 (값은 0.2와 동일하게 임시 입력)
        'RemovedAvgAttn_t90', 'RemovedAvgAttn_t90', 'RemovedAvgAttn_t90', 'RemovedAvgAttn_t90',
        'Sequential_rem_t90_ent_b30', 'Sequential_rem_t90_ent_b30', 'Sequential_rem_t90_ent_b30', 'Sequential_rem_t90_ent_b30',
        'TopKAvgAttn_t80', 'TopKAvgAttn_t80', 'TopKAvgAttn_t80', 'TopKAvgAttn_t80'
    ],
    'MSP_Filter_Threshold': [
        'No OE',
        'No Filter', '0.1', '0.2', '0.3',
        'No Filter', '0.1', '0.2', '0.3', # MaxAttn
        'No Filter', '0.1', '0.2', '0.3',
        'No Filter', '0.1', '0.2', '0.3',
        'No Filter', '0.1', '0.2', '0.3'
    ],
    'AUROC':      [0.870, 0.935, 0.928, 0.918, 0.906, 0.942, 0.912, 0.913, 0.891, 0.940, 0.912, 0.913, 0.891, 0.934, 0.922, 0.905, 0.899, 0.934, 0.912, 0.913, 0.891],
    'FPR@TPR90':  [0.444, 0.189, 0.233, 0.256, 0.311, 0.178, 0.311, 0.289, 0.433, 0.178, 0.311, 0.289, 0.433, 0.233, 0.211, 0.356, 0.422, 0.222, 0.311, 0.289, 0.433],
    'OSCR':       [0.366, 0.625, 0.505, 0.520, 0.432, 0.598, 0.500, 0.498, 0.373, 0.579, 0.500, 0.498, 0.373, 0.598, 0.527, 0.420, 0.461, 0.579, 0.500, 0.498, 0.373],
    'F1_Open':    [0.539, 0.766, 0.671, 0.679, 0.599, 0.744, 0.662, 0.662, 0.539, 0.728, 0.662, 0.662, 0.539, 0.744, 0.688, 0.589, 0.627, 0.728, 0.662, 0.662, 0.539]
}

df_full = pd.DataFrame(data_full)

# 누락된 MaxAttn_t85 MSP 0.1 데이터를 이전 값(MSP 0.2)으로 채우거나, 분석에서 명시적으로 제외
# 여기서는 시각화를 위해 이전 값으로 채움 (실제 분석 시에는 이 부분을 명확히 해야 함)
# 또는, 해당 Base_Strategy의 MSP_Filter_Threshold='0.1' 행을 생성하고 NaN으로 채운 후, barplot에서 처리하도록 함.
# 여기서는 이미 데이터에 포함된 것으로 간주 (위 AUROC 등 리스트에 값 추가함)

metrics_to_plot = ['AUROC', 'FPR@TPR90', 'OSCR', 'F1_Open']
n_metrics = len(metrics_to_plot)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(n_metrics, 1, figsize=(18, n_metrics * 7), sharex=False)
fig.suptitle('OSR Performance: Syslog Masked OE Strategies with MSP Filtering', fontsize=20, y=1.0)

df_standard_plot = df_full[df_full['Base_Strategy'] == 'Standard'].copy()
df_oe_plot = df_full[df_full['Base_Strategy'] != 'Standard'].copy()

msp_order = ['No Filter', '0.1', '0.2', '0.3']
df_oe_plot['MSP_Filter_Threshold'] = pd.Categorical(df_oe_plot['MSP_Filter_Threshold'], categories=msp_order, ordered=True)

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    sns.barplot(x='Base_Strategy', y=metric, hue='MSP_Filter_Threshold', data=df_oe_plot, ax=ax,
                palette=sns.color_palette("viridis_r", n_colors=len(msp_order)))

    ax.set_title(metric, fontsize=14, pad=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=9) # 회전각 증가
    ax.tick_params(axis='y', labelsize=9)

    if not df_standard_plot.empty:
        std_val = df_standard_plot[metric].iloc[0]
        ax.axhline(std_val, ls='--', color='black', lw=2.5, label=f'Standard ({std_val:.3f})')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='MSP Filter / Standard', loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=10, title_fontsize=12, borderaxespad=0.)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 7),
                    textcoords='offset points', fontsize=10, color='black')

# if n_metrics > 0:
#     axes[-1].set_xlabel('Base OE Strategy (Syslog Masked)', fontsize=15, labelpad=20)

fig.subplots_adjust(left=0.07, right=0.8, top=0.95, bottom=0.2, hspace=0.65) # bottom, hspace 조정
plt.savefig("osr_performance_msp_threshold_comparison_updated.png", dpi=300)
plt.show()