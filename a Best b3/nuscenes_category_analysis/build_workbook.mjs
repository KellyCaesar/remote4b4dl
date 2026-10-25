import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const requireRuntime = createRequire('C:/Users/Daniel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const {Workbook, SpreadsheetFile} = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
const outputDir = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(outputDir,'workbook_payload.json'),'utf8'));
const wb = Workbook.create();
const summary = wb.worksheets.add('汇总');
const tasks = Object.keys(data.tasks);
const taskSheets = Object.fromEntries(tasks.map(t=>[t,wb.worksheets.add(t)]));
const mapping = wb.worksheets.add('类别与判定说明');
const ink = '#263341', head = '#304B63', gray = '#EDF1F5';

function base(sheet,range) {
  sheet.showGridLines = false;
  sheet.getRange(range).format.font = {name:'Arial',size:10,color:ink};
  sheet.getRange(range).format.verticalAlignment = 'center';
}
function header(sheet,range) {
  sheet.getRange(range).format = {fill:head,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:30};
}
function width(sheet,col,pixels,end=100) {
  sheet.getRange(`${col}1:${col}${end}`).format.columnWidthPx = pixels;
}

for (const [i,task] of tasks.entries()) {
  const sheet=taskSheets[task], rows=data.tasks[task], end=rows.length+1;
  sheet.getRange(`A1:T${end}`).values=[data.headers,...rows];
  base(sheet,`A1:T${end}`);
  sheet.getRange(`A2:T${end}`).format.wrapText=true;
  sheet.getRange(`A2:T${end}`).format.verticalAlignment='top';
  sheet.getRange(`A2:T${end}`).format.rowHeight=90;
  sheet.getRange(`B2:B${end}`).format.numberFormat='0';
  sheet.getRange(`B2:B${end}`).format.horizontalAlignment='right';
  const widths={A:175,B:70,C:640,D:800,E:240,F:210,G:210,H:210,I:230,J:420,K:540,L:200,M:240,N:260,O:100,P:85,Q:440,R:420,S:520,T:600};
  for(const [col,pixels] of Object.entries(widths)) width(sheet,col,pixels,end);
  const table=sheet.tables.add(`A1:T${end}`,true,`Predictions${i+1}`);
  table.showFilterButton=true;
  sheet.getRange(`A2:T${end}`).format.fill='#FFFFFF';
  header(sheet,'A1:T1');
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  sheet.getRange(`P2:P${end}`).conditionalFormats.add('containsText',{text:'是',format:{fill:'#FFF2D5',font:{color:'#895B12'}}});
  console.log(JSON.stringify({event:'sheet_populated',sheet:task,rows:rows.length}));
}

base(summary,'A1:J41');
summary.tabColor=head;
summary.getRange('A2').values=[['B3生成文本与nuScenes分割类别的对应关系']];
summary.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:ink};
summary.getRange('A4').values=[['每条预测独立分析；类别来自预测原文；一个预测可涉及多个类别。']];
summary.getRange('A5').values=[['肯定提及表示文本中的对象/区域关联，不代表真实场景存在或掩码正确。']];
summary.getRange('A4:A5').format.font={name:'Arial',size:10,color:'#5B6874'};
const taskHeader=['任务类别','预测总数','肯定提及条数','肯定提及比例','含Thing','含Stuff','无16类提及','含否定存在','含风险/建议','需复核'];
const taskRows=tasks.map(task=>{
 const s=data.summary.tasks[task];
 return [task,s.total,s.with_positive,s.with_positive/s.total,s.with_positive_thing,s.with_positive_stuff,s.no_official_mention,s.with_negated,s.with_risk,s.review];
});
summary.getRange('A7:J10').values=[taskHeader,...taskRows];
header(summary,'A7:J7');
summary.getRange('A8:J10').format.rowHeight=30;
summary.getRange('B8:J10').format.numberFormat='#,##0';
summary.getRange('D8:D10').format.numberFormat='0.0%';
summary.getRange('A12').values=[['类别频次：每条预测、每个类别只计一次；下面仅统计肯定提及。']];
const classHeader=['评测类别','中文含义','类型','description','temporal_understanding','comprehensive_reasoning','合计','description比例','temporal比例','reasoning比例'];
const classRows=data.classes.map(([id,name,cn,kind])=>{
 const counts=tasks.map(task=>data.summary.classes[name][task]['肯定提及']);
 return [name,cn,kind,...counts,counts.reduce((a,b)=>a+b,0),...counts.map((v,i)=>v/data.summary.tasks[tasks[i]].total)];
});
summary.getRange('A14:J30').values=[classHeader,...classRows];
header(summary,'A14:J14');
summary.getRange('A15:J30').format.rowHeight=26;
summary.getRange('D15:G30').format.numberFormat='#,##0';
summary.getRange('H15:J30').format.numberFormat='0.0%';
summary.getRange('H15:J30').conditionalFormats.add('colorScale',{colors:['#F6F8FA','#B6CCD9','#4E819C'],thresholds:[0,0.5,1]});
summary.getRange('A33').values=[['未提及不等于不存在；否定只适用于原句中对应的对象、修饰语、位置和帧范围。']];
summary.getRange('A34').values=[['原问题和参考答案仅供核对，未用于补全预测中缺失的类别。']];
summary.getRange('A35').values=[['完整判定规则、原始ignore类别和官方依据见“类别与判定说明”。']];
summary.getRange('A37').values=[['源文件']]; summary.getRange('B37').values=[['predictions.json']];
summary.getRange('A38').values=[['SHA-256']]; summary.getRange('B38').values=[[data.summary.source_sha256]];
summary.getRange('A39').values=[['分析日期']]; summary.getRange('B39').values=[[data.summary.date]];
summary.getRange('A33:A35').format.font={name:'Arial',size:10,color:'#5B6874'};
for(const [col,pixels] of Object.entries({A:220,B:205,C:70,D:125,E:165,F:175,G:115,H:140,I:130,J:140})) width(summary,col,pixels,41);
summary.getRange('B38').format.font={name:'Arial',size:9,color:'#5B6874'};

base(mapping,'A1:E55');
mapping.getRange('A1:E17').values=[['评测编号','评测类别','中文含义','类型','对应的原始类别'],...data.classes];
header(mapping,'A1:E1');
mapping.getRange('A2:E17').format.wrapText=true;
mapping.getRange('A2:E17').format.rowHeight=34;
mapping.getRange('A2:A17').format.numberFormat='0';
const notes=[
 ['肯定提及','原文描写或引用对应的对象/区域。道路、车道等通过语义关联到driveable_surface。'],
 ['否定存在','原文否定对应对象/区域在指定范围内存在或可见。no moving pedestrians仅否定正在运动的行人，不能证明整类行人都不存在。'],
 ['条件或可能','if/may/might等引入对象存在的不确定性，或不同类别之间由or连接。'],
 ['泛指或建议','讨论类别相关活动、道路条件、驾驶建议等，不能单独作为具体实例存在证据。'],
 ['未提及','在该条预测中未取得可映射词；不表示真实场景中没有该类。'],
 ['运动否定','No, the barriers remain static仍提及barrier；no movement of pedestrians不否定行人存在。'],
 ['模态作用范围','已描写的行人could cross属于行为不确定；if a pedestrian crosses属于条件对象。'],
 ['generic vehicle','泛称vehicle/vehicles不能直接确定为car；ego vehicle单列为vehicle.ego。'],
 ['SUV和van','依据官方对象定义归入car；pickup归truck。'],
 ['施工区域/设备','construction zone/site不推出工程车辆；equipment/mixer需确认设备形态；明确construction vehicle/excavator可对应工程车辆。'],
 ['骑行者','cyclist关联bicycle；motorcyclist关联motorcycle；没有额外推断一个pedestrian实例。'],
 ['原始ignore类别','vehicle.ego、animal、noise、personal_mobility、stroller、wheelchair、debris、pushable_pullable、bicycle_rack、ambulance、police vehicle、static.other均不计入官方16类。'],
 ['逐条分析方法','确定性词汇、同义词及局部语境规则覆盖全部记录，结合边界案例人工抽查；保留命中词、字符位置、上下文和需复核原因。'],
 ['统计范围','按预测条目统计，不去除重复预测；类别频次为多标签，不是实例数、点数、场景数或分割准确率。'],
 ['原文顺序','每张任务表的类别内序号从1开始，对应源文件该任务predictions数组索引+1。'],
 ['参考字段','原问题与参考答案原样保留为辅助核对字段；类别抽取只依据预测。'],
 ['规则局限','自然语言存在省略、隐含指代及否定范围歧义；需复核项和泛称候选应在后续标注时进一步检查。'],
 ['真实场景核验','当前文件未含scene/sample/instance定位或点级掩码；不能由文本关联结果认定实际场景类别存在或模型分割正确。'],
 ['结果更新','本表是当前输入文件的分析快照，修改预测后需要重新分析。'],
];
mapping.getRange(`A20:B${19+notes.length}`).values=notes;
mapping.getRange(`A20:B${19+notes.length}`).format.wrapText=true;
mapping.getRange(`A20:B${19+notes.length}`).format.rowHeight=48;
mapping.getRange('A41').values=[['官方来源']];
mapping.getRange('A42:B45').values=data.sources;
mapping.getRange('A42:B45').format.rowHeight=40;
mapping.getRange('A42:B45').format.wrapText=true;
for(const [col,pixels] of Object.entries({A:130,B:720,C:200,D:90,E:720})) width(mapping,col,pixels,55);
mapping.freezePanes.freezeRows(1);

wb.recalculate();
const inspect=await wb.inspect({kind:'table',range:'汇总!A7:J10',include:'values',tableMaxRows:4,tableMaxCols:10,maxChars:3000});
console.log(inspect.ndjson);
for(const [sheetName,range] of [['汇总','A2:J35'],...tasks.map(t=>[t,'A1:D4']),['类别与判定说明','A1:E17']]) {
  const blob=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(outputDir,`preview_${sheetName}.png`),new Uint8Array(await blob.arrayBuffer()));
  console.log(JSON.stringify({event:'rendered',sheet:sheetName}));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(outputDir,'B3预测_nuScenes类别逐条分析.xlsx'));
console.log(JSON.stringify({event:'exported',path:path.join(outputDir,'B3预测_nuScenes类别逐条分析.xlsx')}));
