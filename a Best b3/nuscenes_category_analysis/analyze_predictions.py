"""Auditable text-only nuScenes category association for every generated prediction.

No questions/ground truths are used to infer entities missing from the prediction.
Occurrence-level assertion cues are kept separately from the taxonomy mapping.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "predictions.json"
TASKS = ("description", "temporal_understanding", "comprehensive_reasoning")
STATUSES = ("肯定提及", "否定存在", "条件或可能", "泛指或建议")
CLASSES = [
    (1, "barrier", "隔离栏/临时路障", "Thing", "movable_object.barrier"),
    (2, "bicycle", "自行车", "Thing", "vehicle.bicycle"),
    (3, "bus", "公交车", "Thing", "vehicle.bus.bendy; vehicle.bus.rigid"),
    (4, "car", "汽车/SUV/van", "Thing", "vehicle.car"),
    (5, "construction_vehicle", "工程车辆", "Thing", "vehicle.construction"),
    (6, "motorcycle", "摩托车", "Thing", "vehicle.motorcycle"),
    (7, "pedestrian", "行人", "Thing", "human.pedestrian.adult; human.pedestrian.child; human.pedestrian.construction_worker; human.pedestrian.police_officer"),
    (8, "traffic_cone", "交通锥", "Thing", "movable_object.trafficcone"),
    (9, "trailer", "挂车", "Thing", "vehicle.trailer"),
    (10, "truck", "卡车/皮卡", "Thing", "vehicle.truck"),
    (11, "driveable_surface", "可行驶表面", "Stuff", "flat.driveable_surface"),
    (12, "other_flat", "其他平坦区域", "Stuff", "flat.other"),
    (13, "sidewalk", "人行道/步行道", "Stuff", "flat.sidewalk"),
    (14, "terrain", "自然地表/低矮草地", "Stuff", "flat.terrain"),
    (15, "manmade", "建筑/人工结构", "Stuff", "static.manmade"),
    (16, "vegetation", "树木/灌木/植被", "Stuff", "static.vegetation"),
]
ORDER = {c[1]: c[0] for c in CLASSES}
CN = {c[1]: c[2] for c in CLASSES}
KINDS = {c[1]: c[3] for c in CLASSES}
SOURCES = [
    ("官方16类及ignore映射", "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/utils.py"),
    ("官方Thing/Stuff划分", "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/panoptic/utils.py"),
    ("对象类别定义：SUV/van/施工车辆/骑行者", "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md"),
    ("地面/植被/人工结构定义", "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md"),
]

def rx(s):
    return re.compile(s, re.I)

# Composite labels take priority over embedded nouns (ego car, police car,
# bicycle rack, bus stop). Specific patterns precede broad category patterns.
EXTRA_RULES = [
    ("vehicle.ego", rx(r"\bego[ _-]?(?:vehicles?|cars?)\b")),
    ("vehicle.emergency.police", rx(r"\bpolice\s+(?:vehicles?|cars?|motorcycles?|bicycles?)\b")),
    ("vehicle.emergency.ambulance", rx(r"\bambulances?\b")),
    ("static_object.bicycle_rack", rx(r"\b(?:bicycle|bike)\s+racks?\b")),
    ("human.pedestrian.personal_mobility", rx(r"\b(?:personal mobility (?:devices?|vehicles?)|segways?|skateboards?|kick scooters?)\b")),
    ("human.pedestrian.wheelchair", rx(r"\bwheelchairs?\b")),
    ("human.pedestrian.stroller", rx(r"\b(?:strollers?|baby carriages?|prams?)\b")),
    ("animal", rx(r"\b(?:animals?|dogs?|cats?|birds?|deer|rats?)\b")),
    ("movable_object.debris", rx(r"\b(?:debris|trash bags?|garbage bags?|temporary road[ -]?signs?)\b")),
    ("movable_object.pushable_pullable", rx(r"\b(?:shopping carts?|wheelbarrows?|doll(?:ey|ie)s?|wheeled (?:bins?|garbage bins?))\b")),
    ("noise", rx(r"\b(?:lidar noise|sensor noise|noise points?)\b")),
    ("static.other", rx(r"\b(?:unidentifiable static objects?|unknown static objects?)\b")),
]
RULES = [
    ("manmade", rx(r"\b(?:bus stops?|bus shelters?|traffic lights?|traffic signals?|traffic signs?|street signs?|road signs?|signposts?|lamp[ -]?posts?|street[ -]?lights?|guard[ -]?rails?|railings?|buildings?|walls?|bridges?|fences?|poles?|tunnels?|houses?|hydrants?|parking meters?|artificial structures?|man[ -]?made structures?|static structures?)\b"), "直接词/语义同义词"),
    ("construction_vehicle", rx(r"\b(?:construction[ _-]vehicles?|excavators?|bulldozers?|road rollers?|crane (?:trucks?|vehicles?))\b"), "直接词/语义同义词"),
    ("truck", rx(r"\b(?:concrete mixer trucks?|dump trucks?|semi[ -]trailer trucks?|semi[ -]tractors?|pickup(?: trucks?)?|pick[ -]up trucks?|trucks?|lorr(?:y|ies))\b"), "直接词/语义同义词"),
    ("car", rx(r"\b(?:cars?|SUVs?|vans?|mini[ -]?vans?|sedans?|jeeps?|taxis|taxi|hatchbacks?|automobiles?)\b"), "直接词/语义同义词"),
    ("bus", rx(r"\b(?:buses|bus|coaches|coach|shuttle buses)\b"), "直接词/语义同义词"),
    ("motorcycle", rx(r"\b(?:(?:motorcycle|motorbike) riders?|motorcycles?|motorbikes?|motorcyclists?)\b"), "直接词/骑行者关联"),
    ("bicycle", rx(r"\b(?:bicycles?|cyclists?|bicyclists?|bicycle riders?)\b"), "直接词/骑行者关联"),
    ("pedestrian", rx(r"\b(?:pedestrians?|human[ _.-]pedestrian(?:[ _.-](?:adults?|child|construction[ _.-]worker|police[ _.-]officer))?|construction workers?|police officers?|persons?|people|humans?|walking (?:persons?|people)|children|child)\b"), "直接词/语义同义词"),
    ("barrier", rx(r"\b(?:barriers?|barricades?|roadblocks?)\b"), "直接词/语义同义词"),
    ("traffic_cone", rx(r"\b(?:(?:traffic[ _-]?|construction |safety |road )?cones?|trafficcone)\b"), "直接词/语义同义词"),
    ("trailer", rx(r"\b(?:trailers?|semi[ -]trailers?)\b"), "直接词/语义同义词"),
    ("vegetation", rx(r"\b(?:trees?|bushes|bush|shrubs?|vegetation|plants?|tall grass)\b"), "直接词/语义同义词"),
    ("terrain", rx(r"\b(?:terrain|grassy (?:areas?|surfaces?|patches?)|grass|soil|sand|gravel)\b"), "语义关联，草地高度需场景核对"),
    ("other_flat", rx(r"\b(?:other[ _-]flat|other flat surfaces?|lakes?|rivers?|bodies of water|rail tracks?)\b"), "直接词/语义关联"),
    ("sidewalk", rx(r"\b(?:sidewalks?|footpaths?|pedestrian walkways?|walkways?|bike paths?|bicycle paths?)\b"), "直接词/语义同义词"),
    ("driveable_surface", rx(r"\b(?:driv(?:e)?able[ _-](?:surfaces?|areas?)|roads?|roadways?|streets?|lanes?|intersections?|crosswalks?|road markings?|parking (?:lots?|areas?))\b"), "道路/车道语义关联"),
]
AMBIGUOUS_RULES = [
    ("vehicle泛称", rx(r"\b(?:vehicles?|automotive objects?)\b"), "car/bus/truck/trailer/construction_vehicle/motorcycle/bicycle；不能确定具体类"),
    ("obstacle/object泛称", rx(r"\b(?:obstacles?|objects?)\b"), "对象类型未说明；不能直接推出barrier等类别"),
    ("施工区域", rx(r"\b(?:construction (?:zones?|areas?|sites?|scenes?|elements?)|roadwork)\b"), "施工情境；不能直接推出construction_vehicle"),
    ("施工设备", rx(r"\b(?:(?:construction|heavy) (?:equipment|machinery)|equipment|machinery|concrete mixers?|cement mixers?|cranes?)\b"), "construction_vehicle/truck/manmade或原始ignore类；需确认设备形态"),
    ("structure泛称", rx(r"\b(?:structures?|fixed objects?|static objects?)\b"), "人工结构可关联manmade，但泛称本身不足以确认具体分割类"),
    ("bike/rider/scooter泛称", rx(r"\b(?:bikes?|riders?|scooters?)\b"), "bicycle/motorcycle或human.pedestrian.personal_mobility；需确认交通工具类型"),
    ("ground/pavement泛称", rx(r"\b(?:ground|pavement|curbs?)\b"), "driveable_surface/sidewalk/terrain/other_flat；需确认表面用途"),
    ("仅环境描述", rx(r"\b(?:urban|suburban|cityscape|environment|construction zone)\b"), "环境或场景属性，不独立对应一个分割类"),
]
NEG_PREFIX = rx(r"\b(?:no|without|absence of|lack of|neither|not any)\s+(?:(?:visible|dynamic|moving|stationary|static|nearby|other|any|unexpected|small|smaller|large|significant|immediate|clear|detectable|apparent|additional|specific|physical)\s+)*(?:(?:objects?|obstacles?)\s+(?:like|such as)\s+)?$")
NEG_POST = rx(r"^\s*(?:and\s+(?:other\s+)?\w+\s*)?(?:(?:is|are|was|were)\s+(?:not\s+(?:(?:currently|clearly|readily|immediately)\s+)?(?:visible|present|seen|detected|observed|there)|absent|missing)|(?:cannot|can't|could not)\s+be\s+(?:seen|detected)|(?:do|does)\s+not\s+(?:appear|exist)|(?:are|is)\s+nowhere)")
NOUN_NEG = rx(r"\b(?:no|without|absence of|lack of|neither|not any)\b")
ACTION_OR_RISK_NEG = rx(r"\b(?:changes?|movements?|motion|activity|risks?|threats?|hazards?|danger|issues?|challenges?|need|difference|displacement|variation|speed|collision|impact|effects?|concerns?|importance|change in|changes in)\b")
INDEPENDENT_BREAK = rx(r"[,;]|\b(?:but|while|whereas|although|with|because|causing|indicating|suggesting)\b|\band\s+(?=(?:the|a|an)\s)")
EXISTENCE_CUE = rx(r"\b(?:visible|present|seen|detected|observed|shows?|captures?|includes?|featur(?:es|ing)|there\s+(?:is|are)|presence of|moving|moves?|approach\w*|parked|stationary|remain\w*|positioned|cross(?:es|ing)|located|on the|in front|behind|ahead)\b")
ADVICE = rx(r"\b(?:should|must|need(?:s)? to|requires?|requiring|be aware|be cautious|caution|ensure|ensuring|safe navigation|careful navigation|potential hazards?|risks?|danger|collision)\b")
FRAME = rx(r"\b(?:between\s+)?frames?\s+\d+(?:\s*(?:to|and|[-–])\s*(?:frames?\s+)?\d+)?|\bfrom\s+frame\s+\d+\s+to\s+frame\s+\d+")
SPATIAL = rx(r"\b(?:front[ _-]left|front[ _-]right|back[ _-]left|back[ _-]right|front|back|rear|left|right|ahead|behind|sidewalk|sidewalks|both sides|opposite direction|distance|closer|further away|nearby|adjacent|centrally|center|on the sides)\b")
MOTION = [
    ("靠近/距离缩短", rx(r"\b(?:approach\w*|closer|decreas\w*\s+(?:in\s+)?distance|distance\s+(?:decreas\w*|reduces?))\b")),
    ("远离/距离增大", rx(r"\b(?:further away|moving away|moves away|move away|more distant|receding|increas\w*\s+(?:in\s+)?distance)\b")),
    ("横穿/横向移动", rx(r"\b(?:cross(?:es|ing|ed)?|left to right|right to left|lateral movement)\b")),
    ("转向/换道", rx(r"\b(?:turn\w*|lane changes?|chang\w* lanes|merg\w*|overtak\w*)\b")),
    ("静止/停放", rx(r"\b(?:stationary|static|parked|stopped|remain\w* (?:constant|stable|in place)|no (?:significant )?movement)\b")),
    ("向前/向后运动", rx(r"\b(?:forward|backward|moving|moves|move|walking|travel\w*)\b")),
    ("速度/停车", rx(r"\b(?:speed|slow\w*|accelerat\w*|decelerat\w*|stop\w*)\b")),
]

def uniq(items):
    return list(dict.fromkeys(items))

def overlapping(start, end, spans):
    return any(start < b and end > a for a, b in spans)

def sentence_bounds(text, start, end):
    # Decimal numbers are not present in the frame-based predictions. Dots within
    # category names are ignored unless followed by whitespace/end-of-string.
    bounds = [(m.start(), m.end()) for m in re.finditer(r"[.!?;](?:\s+|$)", text)]
    left = max((b for a, b in bounds if b <= start), default=0)
    right = min((a for a, b in bounds if a >= end), default=len(text))
    return left, right

def assertion(text, start, end):
    left, right = sentence_bounds(text, start, end)
    sentence = text[left:right]
    rel = start - left
    prefix = sentence[:rel]
    suffix = sentence[end-left:]
    # Response-leading "No," answers the question and is not entity negation.
    prefix = re.sub(r"^\s*(?:The answer is\s+)?No,\s*", "", prefix, flags=re.I)
    breaks = list(INDEPENDENT_BREAK.finditer(prefix))
    local = prefix[breaks[-1].end():] if breaks else prefix
    # "road hazards/road conditions/pedestrian activity" names an attribute or
    # topic; absence of that attribute cannot establish absence of its class.
    if rx(r"^\s+(?:hazards?|conditions?|safety|activity|behavior|density|discipline|restrictions?|closures?|users?|crossings)\b").search(suffix) or (NOUN_NEG.search(prefix) and rx(r"^\s+changes?\b").search(suffix)):
        return "泛指或建议", sentence.strip(), "仅讨论类别相关属性/情境"
    negatives = list(NOUN_NEG.finditer(local))
    if negatives:
        between = local[negatives[-1].end():]
        # A negated action/risk does not imply an absent entity.
        if not ACTION_OR_RISK_NEG.search(between) and not rx(r"\b(?:on|along|in|from|towards|to|with|through|between|over|under|near|behind|ahead|allowing|requiring|providing|indicating|suggesting)\b").search(between):
            return "否定存在", sentence.strip(), "名词前的否定存在结构"
    # Long coordinated noun lists: "no pedestrians, animals, or vehicles".
    all_negs = list(NOUN_NEG.finditer(prefix))
    if all_negs:
        between = prefix[all_negs[-1].end():]
        if len(between) < 115 and not ACTION_OR_RISK_NEG.search(between) and not rx(r"\b(?:is|are|was|were|with|while|but|visible|detected|seen|remain\w*|on|along|in|from|towards|to|through|between|over|under|near|behind|ahead|allowing|requiring|providing|indicating|suggesting)\b|\band\s+(?:the|a|an)\s").search(between):
            return "否定存在", sentence.strip(), "并列名词中的否定作用域"
    if NEG_POST.search(suffix):
        return "否定存在", sentence.strip(), "名词后的不可见/不存在谓语"
    # "No dynamic objects like pedestrians are detected".
    if rx(r"^\s*(?:or\s+[\w -]+\s*)?(?:are|is)\s+(?:not\s+)?(?:visible|detected|seen)").search(suffix) and all_negs:
        between = prefix[all_negs[-1].end():]
        if len(between) < 100 and not ACTION_OR_RISK_NEG.search(between):
            return "否定存在", sentence.strip(), "否定的对象集合"
    # Hypothetical existence is distinct from an existing object's uncertain action.
    if rx(r"\bif\s+(?:a|an|any|some|there\s+(?:is|are))\s+(?:\w+\s+){0,4}$").search(prefix):
        return "条件或可能", sentence.strip(), "if引入的条件对象"
    if rx(r"\b(?:could|might|may|would|possibly|potentially)\s+(?:be|have|include|contain|encounter|appear|see)\s+(?:(?:a|an|the|some|any)\s+)?(?:\w+\s+){0,3}$").search(local):
        return "条件或可能", sentence.strip(), "可能存在的对象"
    if rx(r"\b(?:potential|possible|unexpected)\s+(?:hazards?|obstacles?|objects?)\s+(?:like|such as|including)\s+(?:\w+[ ,]+){0,4}$").search(local):
        return "条件或可能", sentence.strip(), "假设风险中的例举对象"
    if rx(r"\bpotential for\s+(?:obstacles?|objects?)\s+(?:like|such as)\s+(?:\w+[ ,]+){0,4}$").search(local):
        return "条件或可能", sentence.strip(), "潜在对象举例"
    # Bare references to activity do not assert an actual instance.
    if rx(r"^\s+(?:activity|movement|behavior|density)\b").search(suffix) and not EXISTENCE_CUE.search(sentence):
        return "泛指或建议", sentence.strip(), "仅讨论类别活动/行为"
    if rx(r"^\s*(?:The answer is\s+)?(?:The )?driver\s+(?:should|must|needs?|has to)|^\s*(?:Drivers?\s+)?(?:should|must)\b").search(sentence):
        if not rx(r"\b(?:visible|present|seen|detected|observed|presence of|approach\w*|parked|stationary|positioned|crossing|moving|walking|proximity|navigate around|on (?:the )?(?:left|right|sidewalks?)|ahead|behind|in front)\b").search(sentence):
            return "泛指或建议", sentence.strip(), "驾驶建议中的泛指对象"
    # A clear subject ("the bus could move") still asserts the bus as a referent;
    # motion modality is captured separately and does not negate the noun.
    return "肯定提及", sentence.strip(), "直接描写/关系描述中的对象或区域"

def extract(text):
    occupied = []
    mentions = []
    extras = []
    for name, pattern in EXTRA_RULES:
        for m in pattern.finditer(text):
            if overlapping(m.start(), m.end(), occupied):
                continue
            status, context, rationale = assertion(text, m.start(), m.end())
            extras.append(dict(category=name, word=m.group(), status=status,
                               span=[m.start(),m.end()], context=context, rationale=rationale))
            occupied.append(m.span())
    for name, pattern, method in RULES:
        for m in pattern.finditer(text):
            if overlapping(m.start(), m.end(), occupied):
                continue
            # "lane change" is an action phrase; it does not independently
            # establish a road region. Same applies to "roadwork" (word boundary).
            if name == "driveable_surface" and m.group().lower() in ("lane", "lanes") and rx(r"^\s+chang").search(text[m.end():]):
                continue
            status, context, rationale = assertion(text, m.start(), m.end())
            mentions.append(dict(category=name, word=m.group(), status=status,
                                 span=[m.start(),m.end()], context=context,
                                 rationale=rationale, mapping=method))
            occupied.append(m.span())
    mentions.sort(key=lambda v:v["span"][0])
    # A disjunction between distinct categories cannot establish both as present.
    for first, second in zip(mentions, mentions[1:]):
        between = text[first["span"][1]:second["span"][0]]
        if first["category"] != second["category"] and rx(r"^\s*,?\s*(?:or|and/or)\s+(?:(?:a|an|the|other)\s+)?$").fullmatch(between):
            if first["status"] != "否定存在" and second["status"] != "否定存在":
                first["status"] = second["status"] = "条件或可能"
                first["rationale"] = second["rationale"] = "不同类别由or连接，类别选择不确定"
    uncertain = []
    for label, pattern, candidates in AMBIGUOUS_RULES:
        for m in pattern.finditer(text):
            if overlapping(m.start(),m.end(),occupied):
                continue
            status, context, rationale = assertion(text,m.start(),m.end())
            uncertain.append(dict(label=label, word=m.group(), status=status,
                                  candidates=candidates, span=list(m.span()), context=context))
    # A construction-equipment mention followed by an explicit excavator in the
    # same sentence is resolved by that named entity; retain its generic wording
    # as context but do not mark the already-resolved occurrence for review.
    for item in uncertain:
        item["resolved"] = item["label"] == "施工设备" and any(
            v["category"] == "construction_vehicle" and v["context"] == item["context"]
            for v in mentions)
    states = {name:[] for _,name,*_ in CLASSES}
    for m in mentions:
        if m["status"] not in states[m["category"]]:
            states[m["category"]].append(m["status"])
    return mentions, extras, uncertain, states

HEADERS = ["任务类别", "类别内序号", "预测原文", "逐条类别分析", "文本肯定提及类别", "文本否定存在类别", "条件或可能类别", "泛指或建议类别", "原始ignore类别", "类别不明的对象或情境", "类别匹配证据", "帧范围原文", "空间关系词", "运动及状态", "风险或驾驶建议", "需复核", "复核原因", "未提及的16类", "原问题", "参考答案"]

def sort_names(names):
    return sorted(names,key=ORDER.__getitem__)

def compact_list(names):
    return "；".join(names) or "无"

def make_record(task,index,text,question,gt):
    mentions, extras, uncertain, states = extract(text)
    by_status = {s:sort_names([n for n,v in states.items() if s in v]) for s in STATUSES}
    positive = by_status["肯定提及"]
    things = [n for n in positive if KINDS[n]=="Thing"]
    stuff = [n for n in positive if KINDS[n]=="Stuff"]
    reasons = []
    for name,v in states.items():
        if "肯定提及" in v and "否定存在" in v:
            reasons.append(f"{name}同时有肯定与否定语句，需按对象/帧范围核对")
        if "条件或可能" in v:
            reasons.append(f"{name}的存在或类别含条件/可能性")
    for u in uncertain:
        if u["label"] in ("施工设备","bike/rider/scooter泛称","ground/pavement泛称") and not u["resolved"]:
            reasons.append(f"{u['word']}：{u['candidates']}")
    if not mentions:
        reasons.append("预测未明确给出官方16类；不能借用问题或参考答案补全")
    if any(m["mapping"].startswith("语义关联，草地") for m in mentions):
        reasons.append("草地/地表按terrain语义关联；高度和真实点标签未核对")
    if by_status["泛指或建议"]:
        reasons.append("泛指或驾驶建议不能单独作为实例存在证据")
    if any(m["word"].lower() in ("cyclist","cyclists","bicyclist","bicyclists","motorcyclist","motorcyclists") for m in mentions):
        reasons.append("骑行者关联到bicycle/motorcycle，未额外推断pedestrian")
    frames = uniq(m.group() for m in FRAME.finditer(text))
    space = uniq(m.group().lower() for m in SPATIAL.finditer(text))
    movement = [n for n,p in MOTION if p.search(text)]
    risk = "是" if ADVICE.search(text) else "否"
    if things and stuff:
        relation = "对象与环境区域"
    elif things:
        relation = "仅对象类别"
    elif stuff:
        relation = "仅环境区域"
    elif mentions:
        relation = "仅否定/条件/泛指类别"
    elif extras:
        relation = "仅原始ignore类别"
    else:
        relation = "无明确分割类别"
    analysis = []
    if things: analysis.append("对象："+"、".join(f"{n}（{CN[n]}）" for n in things))
    if stuff: analysis.append("区域："+"、".join(f"{n}（{CN[n]}）" for n in stuff))
    for status,prefix in [("否定存在","否定存在"),("条件或可能","条件/可能"),("泛指或建议","泛指/建议")]:
        if by_status[status]: analysis.append(prefix+"："+"、".join(by_status[status]))
    if extras:
        extra_names = uniq(f"{m['category']}（{m['status']}）" for m in extras)
        analysis.append("原始ignore："+"、".join(extra_names))
    if not mentions: analysis.append("未明确给出官方16类，无法仅从预测确定具体分割类别")
    hard_uncertain = uniq(u["word"] for u in uncertain if u["label"] not in ("仅环境描述",) and not u["resolved"])
    if hard_uncertain: analysis.append("泛称/待定："+"、".join(hard_uncertain))
    if frames: analysis.append("时间："+"、".join(frames))
    if movement: analysis.append("关系/属性："+"、".join(movement))
    if risk=="是": analysis.append("含风险判断/驾驶建议；该部分不是分割类别")
    evidence = uniq(f"{m['category']}←{m['word']}[{m['status']}]" for m in mentions)
    row = [task,index,text,"；".join(analysis),*(compact_list(by_status[s]) for s in STATUSES),
           compact_list(uniq(f"{m['category']}[{m['status']}]" for m in extras)),
           compact_list(uniq(f"{u['word']}→{u['candidates']}" for u in uncertain if u["label"]!="仅环境描述")),
           compact_list(evidence),compact_list(frames),compact_list(space),compact_list(movement),risk,
           "是" if reasons else "否",compact_list(uniq(reasons)),
           compact_list(n for _,n,*_ in CLASSES if not states[n]),question,gt]
    return dict(task=task,index=index,prediction=text,row=row,mentions=mentions,extras=extras,
                uncertain=uncertain,states=states,positive_things=things,positive_stuff=stuff,
                relation=relation,frames=frames,movement=movement,risk=risk,
                review_reasons=uniq(reasons))

def self_check():
    # Contrastive, meaningful language cases for the requested presence analysis.
    cases = [
        ("No, the barriers remain static.", {"barrier":["肯定提及"]}),
        ("There are no pedestrians, animals, or other vehicles nearby.", {"pedestrian":["否定存在"]}),
        ("The pedestrians do not move.", {"pedestrian":["肯定提及"]}),
        ("There is no significant movement from the pedestrians.", {"pedestrian":["肯定提及"]}),
        ("If a pedestrian crosses the road, the driver should stop.", {"pedestrian":["条件或可能"]}),
        ("The pedestrian is visible on the sidewalk and could cross the road.", {"pedestrian":["肯定提及"]}),
        ("The ego vehicle moves forward.", {"car":[]}),
        ("A bus stop is next to a bicycle rack.", {"bus":[],"bicycle":[],"manmade":["肯定提及"]}),
        ("A construction zone is visible.", {"construction_vehicle":[]}),
        ("A construction vehicle and an SUV are parked.", {"construction_vehicle":["肯定提及"],"car":["肯定提及"]}),
        ("A car or truck is ahead.", {"car":["条件或可能"],"truck":["条件或可能"]}),
        ("A truck is stationary or moving slowly.", {"truck":["肯定提及"]}),
        ("Yes.", {"car":[],"pedestrian":[]}),
        ("There are no pedestrians on the road.", {"pedestrian":["否定存在"],"driveable_surface":["肯定提及"]}),
        ("No significant road hazards are noted.", {"driveable_surface":["泛指或建议"]}),
        ("The road is clear with no immediate obstacles, indicating a route through the urban street.", {"driveable_surface":["肯定提及"]}),
        ("The pedestrian's movement across the road requires the driver to be cautious.", {"pedestrian":["肯定提及"],"driveable_surface":["肯定提及"]}),
        ("The scene is static with no pedestrians or moving vehicles, and the road is clear.", {"pedestrian":["否定存在"],"driveable_surface":["肯定提及"]}),
        ("No significant terrain changes are observed.", {"terrain":["泛指或建议"]}),
        ("No detected active road users are present.", {"driveable_surface":["泛指或建议"]}),
        ("A pedestrian crossing the road is visible.", {"pedestrian":["肯定提及"]}),
        ("The car changes lanes.", {"car":["肯定提及"]}),
    ]
    for text,expected in cases:
        states = extract(text)[3]
        for name,want in expected.items():
            assert states[name] == want,(text,name,states[name],want)

def write_csv(path,headers,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(headers)
        w.writerows(rows)

def main():
    self_check()
    raw=SOURCE.read_bytes()
    source=json.loads(raw.decode("utf-8-sig"))
    records=[]
    for task in TASKS:
        group=source[task]
        assert len(group["predictions"])==len(group["questions"])==len(group["ground_truths"])
        assert all(isinstance(s,str) for s in group["predictions"])
        for i,(p,q,gt) in enumerate(zip(group["predictions"],group["questions"],group["ground_truths"]),1):
            records.append(make_record(task,i,p,q,gt))
    summary=dict(source_file=str(SOURCE),source_sha256=hashlib.sha256(raw).hexdigest(),
                 protocol="text-only-taxonomy-association-v1",date="2026-10-12",tasks={},classes={},
                 total=len(records),sources=SOURCES)
    task_rows=[]
    for task in TASKS:
        rs=[r for r in records if r["task"]==task]
        counts=dict(total=len(rs),unique_predictions=len({r["prediction"] for r in rs}),
                    with_official_mention=sum(bool(r["mentions"]) for r in rs),
                    with_positive=sum(any("肯定提及" in v for v in r["states"].values()) for r in rs),
                    with_positive_thing=sum(bool(r["positive_things"]) for r in rs),
                    with_positive_stuff=sum(bool(r["positive_stuff"]) for r in rs),
                    with_both=sum(bool(r["positive_things"] and r["positive_stuff"]) for r in rs),
                    with_negated=sum(any("否定存在" in v for v in r["states"].values()) for r in rs),
                    with_conditional=sum(any("条件或可能" in v for v in r["states"].values()) for r in rs),
                    with_generic=sum(any("泛指或建议" in v for v in r["states"].values()) for r in rs),
                    with_frames=sum(bool(r["frames"]) for r in rs),
                    with_risk=sum(r["risk"]=="是" for r in rs),
                    review=sum(bool(r["review_reasons"]) for r in rs),
                    no_official_mention=sum(not r["mentions"] for r in rs))
        summary["tasks"][task]=counts
        summary["tasks"][task]["relations"]=dict(collections.Counter(r["relation"] for r in rs))
        task_rows.append([task,counts["total"],counts["with_official_mention"],counts["with_positive"],counts["with_positive_thing"],counts["with_positive_stuff"],counts["with_negated"],counts["with_conditional"],counts["with_generic"],counts["with_frames"],counts["with_risk"],counts["no_official_mention"],counts["review"]])
        write_csv(ROOT/f"{task}_逐条分析.csv",HEADERS,(r["row"] for r in rs))
    class_rows=[]
    for cid,name,cn,kind,rawnames in CLASSES:
        entry={}
        for task in TASKS:
            rs=[r for r in records if r["task"]==task]
            vals={s:sum(s in r["states"][name] for r in rs) for s in STATUSES}
            vals["任意提及"]=sum(bool(r["states"][name]) for r in rs)
            entry[task]=vals
            class_rows.append([task,cid,name,cn,kind,len(rs),vals["任意提及"],vals["肯定提及"],vals["否定存在"],vals["条件或可能"],vals["泛指或建议"],vals["肯定提及"]/len(rs)])
        summary["classes"][name]=entry
    write_csv(ROOT/"全部预测_逐条类别对应.csv",HEADERS,(r["row"] for r in records))
    write_csv(ROOT/"类别频次汇总.csv",["任务类别","评测类别编号","类别","中文含义","Thing或Stuff","总预测数","任意提及条数","肯定提及条数","否定存在条数","条件或可能条数","泛指或建议条数","肯定提及比例"],class_rows)
    write_csv(ROOT/"16类存在关系矩阵.csv",["任务类别","类别内序号"]+[n for _,n,*_ in CLASSES],([r["task"],r["index"]]+["/".join(r["states"][n]) or "未提及" for _,n,*_ in CLASSES] for r in records))
    write_csv(ROOT/"需复核预测.csv",HEADERS,(r["row"] for r in records if r["review_reasons"]))
    with (ROOT/"逐条分析.jsonl").open("w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    (ROOT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    payload=dict(headers=HEADERS,task_summary_headers=["任务类别","预测总数","任意官方类提及","肯定提及","含肯定Thing","含肯定Stuff","含否定存在","含条件可能","含泛指建议","含帧范围","含风险建议","无官方类提及","需复核"],
                 task_summary=task_rows,class_summary_headers=["任务类别","评测编号","类别","中文含义","类型","预测总数","任意提及","肯定提及","否定存在","条件可能","泛指建议","肯定比例"],
                 class_summary=class_rows,classes=CLASSES,sources=SOURCES,
                 tasks={task:[r["row"] for r in records if r["task"]==task] for task in TASKS},summary=summary)
    (ROOT/"workbook_payload.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"total":len(records),"tasks":summary["tasks"],"sha256":summary["source_sha256"]},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
