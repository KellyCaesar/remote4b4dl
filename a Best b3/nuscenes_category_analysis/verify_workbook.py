from pathlib import Path
import json
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'workbook_payload.json').read_text(encoding='utf8'))
path=ROOT/'B3预测_nuScenes类别逐条分析.xlsx'
wb=load_workbook(path,read_only=False,data_only=False)
assert wb.sheetnames==['汇总','description','temporal_understanding','comprehensive_reasoning','类别与判定说明']
verified=0
for task,expected in data['tasks'].items():
    ws=wb[task]
    assert ws.max_row==len(expected)+1,(task,ws.max_row,len(expected))
    assert ws.max_column==20
    assert [c.value for c in ws[1]]==data['headers']
    assert ws.freeze_panes=='C2',(task,ws.freeze_panes)
    assert len(ws.tables)==1
    assert next(iter(ws.tables.values())).ref==f'A1:T{len(expected)+1}'
    for row,want in zip(ws.iter_rows(min_row=2,values_only=True),expected):
        got=[v if v is not None else '' for v in row]
        normalized=[v if v is not None else '' for v in want]
        assert got==normalized,(task,want[1],[(i,a,b) for i,(a,b) in enumerate(zip(got,normalized)) if a!=b])
        verified+=1
    assert not any(c.data_type=='f' for row in ws for c in row),'Unexpected formula in source/analysis text'
for i,task in enumerate(data['tasks'],8):
    s=data['summary']['tasks'][task]
    want=[task,s['total'],s['with_positive'],s['with_positive']/s['total'],s['with_positive_thing'],s['with_positive_stuff'],s['no_official_mention'],s['with_negated'],s['with_risk'],s['review']]
    actual=[wb['汇总'].cell(i,j).value for j in range(1,11)]
    for a,b in zip(actual,want):
        assert (abs(a-b)<1e-12 if isinstance(b,float) else a==b),(task,actual,want)
assert verified==16067
wb.close()
report=json.loads((ROOT/'verification.json').read_text(encoding='utf8'))
report['xlsx_all_20_columns_verified']=verified
report['xlsx_summary_matches_current_analysis']=True
report['xlsx_filters_and_freeze_panes_verified']=True
report['xlsx_size_bytes']=path.stat().st_size
(ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:report[k] for k in ['xlsx_all_20_columns_verified','xlsx_summary_matches_current_analysis','xlsx_filters_and_freeze_panes_verified','xlsx_size_bytes']},ensure_ascii=False))
