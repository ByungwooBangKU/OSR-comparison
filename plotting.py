import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 새로운 실험 데이터
data_new = {
    'Strategy': [
        'Standard (No OE)',
        'Entropy (t80)', # attention_entropy_top80pct
        'MaxAttn (b20)', # max_attention_bottom20pct
        'RemovedAvgAttn (t80)', # removed_avg_attention_top80pct
        'Sequential (rem_h80_ent_h75)', # sequential_removed_avg_attention_higher_80_attention_entropy_higher_75
        'TopKAvgAttn (b20)' # top_k_avg_attention_bottom20pct
    ],
    'AUROC':              [0.869841, 0.936640, 0.855247, 0.966534, 0.941534, 0.866314],
    'FPR@TPR90':          [0.444444, 0.188889, 0.511111, 0.088889, 0.100000, 0.466667],
    'AUPR_In':            [0.951382, 0.977873, 0.949959, 0.987887, 0.980373, 0.953621],
    'AUPR_Out':           [0.681159, 0.820633, 0.595871, 0.911956, 0.812519, 0.621837],
    'DetectionAccuracy':  [0.809942, 0.853801, 0.766082, 0.923977, 0.850877, 0.769006],
    'OSCR':               [0.365690, 0.510042, 0.218131, 0.737424, 0.498001, 0.219805],
    'Closed_Set_Accuracy':[0.837302, 0.833333, 0.837302, 0.833333, 0.833333, 0.813492],
    'F1_Macro':           [0.734364, 0.731966, 0.722016, 0.733084, 0.736756, 0.689042]
}
df_new = pd.DataFrame(data_new)

# 시각화할 지표들
metrics_to_plot_new = ['AUROC', 'FPR@TPR90', 'OSCR', 'DetectionAccuracy', 'AUPR_Out', 'Closed_Set_Accuracy', 'F1_Macro']
n_metrics_new = len(metrics_to_plot_new)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(n_metrics_new, 1, figsize=(15, n_metrics_new * 6), sharex=True)
fig.suptitle('OSR Performance Comparison of New OE Strategies (vs Standard)', fontsize=20, y=1.0)

df_standard_plot = df_new[df_new['Strategy'] == 'Standard (No OE)'].copy()
df_oe_strategies_plot = df_new[df_new['Strategy'] != 'Standard (No OE)'].copy()


for i, metric in enumerate(metrics_to_plot_new):
    ax = axes[i]
    
    sns.barplot(x='Strategy', y=metric, data=df_oe_strategies_plot, ax=ax, palette="viridis")

    ax.set_title(metric, fontsize=10, pad=12)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=9)
    ax.tick_params(axis='y', labelsize=9)

    if not df_standard_plot.empty and metric in df_standard_plot.columns:
        std_val = df_standard_plot[metric].iloc[0]
        ax.axhline(std_val, ls='--', color='red', lw=2, label=f'Standard ({std_val:.3f})')
        
        # --- 범례 위치 수정 ---
        # Standard 라인에 대한 범례만 있으므로, 핸들과 레이블을 직접 가져와서 설정
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                  loc='upper left', bbox_to_anchor=(1.01, 1), # 오른쪽 바깥으로 이동
                  fontsize=7, title_fontsize=11, borderaxespad=0.)
        # --- 수정 끝 ---

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 7),
                    textcoords='offset points', fontsize=9, color='black')

# if n_metrics_new > 0:
#     axes[-1].set_xlabel('OE Strategy (Syslog Masked)', fontsize=15, labelpad=20)

# 범례를 바깥으로 뺐으므로, 오른쪽 여백을 더 확보해야 합니다.
fig.subplots_adjust(left=0.08, right=0.85, top=0.93, bottom=0.11, hspace=0.6) # right 값을 줄여서 범례 공간 확보

plt.savefig("osr_performance_new_strategies_legend_outside.png", dpi=300)
plt.show()