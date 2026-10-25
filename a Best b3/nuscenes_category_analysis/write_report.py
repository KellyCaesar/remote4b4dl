from pathlib import Path
import collections
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent

def main():
    from analyze_predictions import CLASSES,TASKS,SOURCE,SOURCES,HEADERS,self_check
    self_check()
    summary=json.loads((ROOT/'summary.json').read_text(encoding='utf8'))
    original=json.loads(SOURCE.read_text(encoding='utf-8-sig'))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==summary['source_sha256']
    records=[json.loads(x) for x in (ROOT/'逐条分析.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(records)==sum(len(original[t]['predictions']) for t in TASKS)==16067
    assert len({(r['task'],r['index']) for r in records})==len(records)
    for r in records:
        assert r['prediction']==original[r['task']]['predictions'][r['index']-1]
        assert r['row'][18]==original[r['task']]['questions'][r['index']-1]
        assert r['row'][19]==original[r['task']]['ground_truths'][r['index']-1]
        assert len(r['row'])==len(HEADERS)
        for m in r['mentions']+r['extras']:
            assert r['prediction'][m['span'][0]:m['span'][1]]==m['word']
        assert set(r['states'])=={x[1] for x in CLASSES}
    for t in TASKS:
        with (ROOT/f'{t}_逐条分析.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.reader(f))
        assert rows[0]==HEADERS
        assert len(rows)-1==len(original[t]['predictions'])
        assert all(row[2]==p for row,p in zip(rows[1:],original[t]['predictions']))
    with (ROOT/'16类存在关系矩阵.csv').open(encoding='utf-8-sig',newline='') as f:
        matrix=list(csv.reader(f))
    assert len(matrix)==16068
    assert all(len(row)==18 for row in matrix)
    for name,per_task in summary['classes'].items():
        for task,counts in per_task.items():
            for status in ('肯定提及','否定存在','条件或可能','泛指或建议'):
                assert counts[status]==sum(status in r['states'][name] for r in records if r['task']==task)
    totals={key:sum(v[key] for v in summary['tasks'].values()) for key in ('total','with_positive','no_official_mention','with_negated','with_conditional','with_generic','review')}
    check=dict(source_sha256=summary['source_sha256'],source_unchanged=True,
               aligned_prediction_rows=len(records),unique_task_index_pairs=len(records),
               csv_rows_verified=True,evidence_spans_verified=True,category_matrix_rows=16067,
               class_frequency_recomputed=True,totals=totals,
               method='全量规则分析及原文/证据/汇总一致性核验；典型和边界案例人工抽查')
    (ROOT/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# B3预测内容与nuScenes分割类别的逐条对应分析','',
           '分析日期：2026-10-12。输入为用户提供的 predictions.json。只分析 description、temporal_understanding 和 comprehensive_reasoning，共 **16,067 条预测**。',
           '', '**一条预测对应结果表的一行，保留任务内原始顺序和重复输出。** 类别内序号从1开始，对应源文件 predictions 数组索引加1。原问题和参考答案仅保留为核对字段，类别提取不借用这些字段。',
           '', '## 判定范围','',
           '分析的是生成文本与类别定义的对应关系。采用可复现的词汇、同义词及局部语境规则覆盖全部预测，人工核对典型和边界表达；每条记录保留命中词、上下文及复核原因。不是人工逐条确认所有自然语言语义，也不是场景存在性或分割准确率评测。',
           '', '官方评测16类为10类Thing和6类Stuff；原始32类中的自车、动物等另记为原始ignore类别。文本中的位置、运动、帧范围和驾驶建议另外列出，不作为新增类别。',
           '', '| 关系状态 | 含义 |', '|---|---|',
           '| 肯定提及 | 文本描写或引用该对象/区域；可包含SUV→car、道路→driveable_surface等语义对应。 |',
           '| 否定存在 | 原文否定对应对象/区域存在或可见，只作用于原句的修饰语、对象、位置和帧范围。 |',
           '| 条件或可能 | 对象存在或类别带条件/可能性，或不同类别由or连接。 |',
           '| 泛指或建议 | 讨论类别活动、路况或驾驶建议，不能单独作为具体实例存在证据。 |',
           '| 未提及 | 没有从预测中取得该类指称；不表示真实场景不存在该类。 |',
           '', '同一预测的同一类别可在不同句子、实例或帧范围内出现多种状态。肯定与否定并存会保留两者并标记需复核，不强行合并。',
           '', '## 三类任务的内容特点','',
           '| 任务 | 条数 | 至少一个肯定类别 | 同时含肯定Thing和Stuff | 无明确16类提及 | 含风险/建议 |',
           '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        v=summary['tasks'][task]
        lines.append(f"| {task} | {v['total']:,} | {v['with_positive']:,}（{v['with_positive']/v['total']:.1%}） | {v['with_both']:,}（{v['with_both']/v['total']:.1%}） | {v['no_official_mention']:,} | {v['with_risk']:,} |")
    lines+=['',
      '**description：对象和环境一起描述。** 常见组合是行人、汽车、公交车、卡车，加上道路、人行道、建筑和树木。其生成内容通常还能给出左右位置、前后距离、运动状态或帧范围；这些是对象的属性与关系。',
      '',
      '**temporal_understanding：更多描述运动和相对变化。** 比如某辆车是否靠近、行人是否横穿、路障是否移动。有些输出仅使用they/it或泛称vehicles，不足以仅从预测确定具体类别；Yes./No.也不会从问题中补全类别。',
      '',
      '**comprehensive_reasoning：对象与驾驶影响、风险判断相联系。** 比如靠近的公交车需要保持距离、路障和交通锥提示施工。存在的对象、假设风险中的例举对象以及泛指建议分开记录，避免把“可能出现交通锥”记成已经存在交通锥。',
      '', '## 官方16类的文本关联频次','',
      '下表按“肯定提及”统计，每条预测、每个类别只计一次。类别可共现，因此各类别计数相加会超过预测条数。百分比的分母为该任务的预测总数，不是点数、实例数或场景数。',
      '', '| 类别 | 类型 | description | temporal_understanding | comprehensive_reasoning |', '|---|---|---:|---:|---:|']
    for _,name,cn,kind,_ in CLASSES:
        vals=[]
        for task in TASKS:
            n=summary['classes'][name][task]['肯定提及']
            vals.append(f"{n:,}（{n/summary['tasks'][task]['total']:.1%}）")
        lines.append(f"| {name}（{cn}） | {kind} | "+' | '.join(vals)+' |')
    lines+=['', '![三类任务的肯定类别提及比例](类别提及比例.png)',
      '', '高频类别集中在pedestrian、car、bus、truck、driveable_surface和sidewalk。construction_vehicle、trailer、bicycle、terrain较少。规则未取得可靠的other_flat指称；这只说明当前生成文本没有清楚提及，不能推断对应场景没有此类点。',
      '', '## 逐条案例','', '| 任务与序号 | 预测原文 | 对应关系 |', '|---|---|---|']
    example_keys=[('description',1),('description',605),('temporal_understanding',5),('temporal_understanding',241),('temporal_understanding',573),('comprehensive_reasoning',7497),('comprehensive_reasoning',5096)]
    for task,index in example_keys:
        r=next(r for r in records if r['task']==task and r['index']==index)
        lines.append(f"| {task} #{index} | {r['prediction'].replace('|','/')} | {r['row'][3].replace('|','/')} |")
    lines+=['', '## 必须保留的类别边界','',
      '- ego vehicle对应原始vehicle.ego，官方评测忽略；不会据此额外增加car。',
      '- SUV和通常所指的van按官方个人用车定义关联car；pickup关联truck。车辆的真实用途仍需场景标注确认。',
      '- construction site/zone只是施工情境，不能推出construction_vehicle。明确工程车辆或excavator可以对应工程车辆；equipment/concrete mixer保持候选关系，避免将静态设备、搅拌车和工程车辆混为一类。',
      '- buildings、walls、traffic lights/signals等关联manmade；不是额外增加“建筑”“红绿灯”评测类。未指明类型的structure保留为泛称。',
      '- cyclist关联bicycle，motorcyclist关联motorcycle；不会自动另加pedestrian实例。',
      '- no movement/No, ... remain static否定的是运动，不是否定对象存在。no moving pedestrians仅否定正在运动的行人，不证明场景中整类行人不存在。',
      '- if a pedestrian crosses是条件对象；已经描述的行人could cross属于行为可能性，应保留已提及对象。',
      '- roadside、lane change、construction、city/urban等不能只按子字符串拆出不存在的类别。草地按terrain语义关联，但点级标注中的高度界限没有由文本验证。',
      '', '## 不明确内容与后续使用','',
      f"共有 **{totals['no_official_mention']:,} 条**预测没有明确提供官方16类指称；其中可包括自车、泛称车辆、施工情境、代词或简短回答。这些记录已完整保留。",
      '', f"共有 **{totals['review']:,} 条**在规则分析中标记了复核原因，包括多种存在状态并存、条件/泛指表达、类型不明设备、草地边界、骑行者关联或缺少具体类别。这个标记表示语义判定的复核需要，不表示模型输出错误。",
      '', '用于后续指代分割数据筛选时，Thing肯定提及可以提供目标类别候选，位置和运动可提供描述约束；Stuff肯定提及可以提供区域候选。仍需把预测关联到真实scene/sample和对象实例或点级区域，才能生成有效的分割监督。当前文件没有提供这种定位或掩码，不能直接从文本确定GT实例、点集、漏分或误分。',
      '', '## 交付文件','',
      '- B3预测_nuScenes类别逐条分析.xlsx：汇总、三个任务的逐条表、类别与判定说明，可筛选。',
      '- 全部预测_逐条类别对应.csv：全部16,067行；另有三个任务各自的CSV。',
      '- 16类存在关系矩阵.csv：每条预测对16类的状态，含未提及。',
      '- 需复核预测.csv：保留复核原因的记录子集。',
      '- 逐条分析.jsonl：原始预测、命中词、字符范围、原句上下文和结构化状态。',
      '', '## 数据核验与复现','',
      '已逐条核对预测、问题、参考答案的顺序与内容；核对证据字符范围能回指预测原文；从逐条记录独立重算各类别频次；核对CSV和16类矩阵的记录数。源文件未修改。',
      '', f"源文件SHA-256：`{summary['source_sha256']}`。重复生成文本仍按原始条目计数；未进行去重或类别均衡处理。",
      '', '## 官方依据','']
    lines += [f'- [{title}]({url})' for title,url in SOURCES]
    (ROOT/'分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=[c[1] for c in CLASSES]
    counts=np.array([[summary['classes'][name][task]['肯定提及'] for task in TASKS] for name in names])
    rates=counts/np.array([summary['tasks'][task]['total'] for task in TASKS])*100
    fig,ax=plt.subplots(figsize=(10,10),dpi=170)
    im=ax.imshow(rates,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(3),[f'{task}\n(n={summary["tasks"][task]["total"]:,})' for task in TASKS],fontsize=9)
    ax.set_yticks(range(16),names,fontsize=10)
    ax.tick_params(length=0,pad=9)
    ax.set_title('Predictions with an affirmative category mention (%)',fontsize=13,pad=18)
    for i in range(16):
        for j in range(3):
            ax.text(j,i,f'{rates[i,j]:.1f}%  ({counts[i,j]:,})',ha='center',va='center',fontsize=9,color='white' if rates[i,j]>60 else '#1f3345')
    ax.axhline(9.5,color='#71869a',lw=1.4)
    for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.025)
    cb.set_label('% of predictions',fontsize=9)
    fig.text(0.01,0.015,'Text associations only; no scene-presence or segmentation-accuracy evaluation.',fontsize=9,color='#526273')
    fig.tight_layout(rect=(0,0.035,1,1))
    fig.savefig(ROOT/'类别提及比例.png',bbox_inches='tight')
    fig.savefig(ROOT/'类别提及比例.svg',bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(check,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
