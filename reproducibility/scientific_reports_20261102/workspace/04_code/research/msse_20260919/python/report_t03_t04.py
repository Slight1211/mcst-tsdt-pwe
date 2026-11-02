from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
a=ROOT/'runs/t03_N13_20260919T195234489663Z';b=ROOT/'runs/t04_X4_20260919T195357266896Z'
r=json.loads((a/'refinement.json').read_text());g=json.loads((a/'geometry.json').read_text());m=json.loads((b/'metadata.json').read_text())
assert m['execution_status']=='completed'
s=json.loads((b/'sensitivity.json').read_text());assert len(s)==8
lines=['# T03加密与T04单模态核验（2026-09-20）','','## T03选定控制点检查','','| 点 | N11→13前12条最大频率变化 | 最低子空间相关度 | 最大簇平均参与率变化 |','|---|---:|---:|---:|']
for x in r:lines.append(f"| {x['point']} | {100*x['max_frequency_change']:.4f}% | {min(z['min_subspace_overlap'] for z in x['clusters']):.8f} | {max(z['mean_participation_change'] for z in x['clusters']):.6f} |")
lines+=['','N13总自由度3645。低阶Fourier基嵌入高阶基后，采用同一高阶质量内积；先全局分配，再比较近简并簇主角度。频率门槛0.5%、参与率门槛0.01未放宽；本次新增身份检查门槛为子空间相关度0.99。两点均通过这些局部检查，不等同于全BZ收敛或独立验证。','','固定N5，圆形32/64/128/256节点几何谱与解析圆比较：']
for x in g:
 if x['nodes']==256:lines.append(f"- {x['point']}点256节点最大频率误差：{100*x['max_frequency_error']:.5f}%。")
lines+=['','## T04小型核验','','只针对MCST-TSDT圆形两固相基线、X点第4条孤立模态。N13，h/a=0.1，两相ell/h=0.25，假设材料参数。先写入不可更改的预测文件，之后才计算ell乘exp(±epsilon)的扰动。epsilon为0.01、0.001、0.0001、0.00001。','','| 相 | 参与率预测 | 四步长最大导数绝对误差 | 最低MAC |','|---|---:|---:|---:|']
for phase in ['A','B']:
 x=[z for z in s if z['phase']==phase];lines.append(f"| {phase} | {x[0]['predicted']:.8f} | {max(z['absolute_error'] for z in x):.3e} | {min(z['min_MAC'] for z in x):.8f} |")
lines+=['','这证明该离散模型在这个简单模态上满足相内能量—灵敏度身份式，不能作为新颖性或带隙设计成功的证据。相同百分比内尺度扰动下，该模态对B相比A相更敏感；不推广到其他模态或全频带。','','## 仍未完成','','- 两相四模型完整对照、内部BZ极值及全频带适用性。','- 几何变动后的严格场匹配及参与率几何收敛。','- 多模态选择性、等预算空间干预、固定材料优化。','- 独立IGA或三维同模型复核。','- 未取得文献的全文核对。','','原论文/PDF没有覆盖，本轮是计算证据补充。原始文件分别见：',str(a),str(b),'','实际入口：python/check_t03_final.py、python/check_phase_prediction.py。运行环境沿用Python3.12.14、NumPy2.5.2、SciPy1.18.1；源码哈希和配置保存在各运行目录。']
(ROOT/'REPORT_T03_T04_20260920.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
