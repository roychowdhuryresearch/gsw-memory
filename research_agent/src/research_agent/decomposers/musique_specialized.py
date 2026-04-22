import re
from typing import List, Optional
from research_agent.decomposers._schema import SubQuestion

class SubIdGen:
    def __init__(self):
        self.counter = 0
    def next(self) -> str:
        self.counter += 1
        return f's{self.counter}'

relation_keywords = {
    'place of birth': 'place of birth',
    'birthplace': 'place of birth',
    'born in': 'place of birth',
    'born': 'place of birth',
    'place of death': 'place of death',
    'died in': 'place of death',
    'died': 'place of death',
    'performer of': 'performer',
    'performer': 'performer',
    'author of': 'author',
    'author': 'author',
    'director of': 'director',
    'director': 'director',
    'screenwriter of': 'screenwriter',
    'screenwriter': 'screenwriter',
    'child of': 'child',
    'child': 'child',
    'sibling': 'sibling',
    'sister': 'sibling',
    'brother': 'sibling',
    'spouse': 'spouse',
    'publisher': 'publisher',
    'formed': 'formed',
    'created': 'created',
    'ended': 'ended',
    'abolished': 'abolished',
    'educated at': 'educated at',
    'located in': 'location',
    'location': 'location',
    'headquarters location': 'headquarters location',
    'headquarters of': 'headquarters location',
    'number of episodes': 'number of episodes',
    'mascot': 'mascot',
    'operator': 'operator',
    'succeeded': 'followed by',
    'followed by': 'followed by',
    'singer of': 'performer',
    'singer': 'performer',
    'song by': 'performer',
    'song': 'song',
    'album': 'album',
    'band': 'band',
    'group': 'group',
    'season': 'season',
    'episode': 'episode',
    'date of birth': 'date of birth',
    'birth date': 'date of birth',
    'date of death': 'date of death',
    'date': 'date',
    'year': 'year',
    'month': 'month',
    'day': 'day',
    'time': 'time',
    'start date': 'start date',
    'end date': 'end date',
    'head of': 'head of',
    'mascot of': 'mascot',
    'songwriter of': 'songwriter',
    'composer of': 'composer',
    'composer': 'composer',
    'producer of': 'producer',
    'producer': 'producer',
    'recorded at': 'location',
    'recorded in': 'location',
    'located at': 'location',
    'capital of': 'capital',
    'capital city of': 'capital',
    'capital': 'capital',
    'region': 'region',
    'county': 'county',
    'state': 'state',
    'country': 'country',
    'continent': 'continent',
    'city': 'city',
    'town': 'town',
    'village': 'village',
    'river': 'river',
    'lake': 'lake',
    'sea': 'sea',
    'ocean': 'ocean',
    'battle': 'battle',
    'war': 'war',
    'conflict': 'conflict',
    'episodes': 'number of episodes',
    'episode count': 'number of episodes',
    'number of seasons': 'number of seasons',
    'season count': 'number of seasons',
    'record company': 'record label',
    'record label': 'record label',
    'label': 'record label',
    'single': 'song',
    'lyricist': 'lyricist',
    'actor': 'actor',
    'actress': 'actor',
    'musician': 'performer',
    'orchestra': 'group',
    'choir': 'group',
    'conductor': 'conductor',
    'arranger': 'arranger',
    'engineer': 'engineer',
    'mixer': 'mixer',
    'mastering engineer': 'mastering engineer',
    'distributor': 'distributor',
    'studio': 'studio',
    'venue': 'venue',
    'festival': 'festival',
    'award': 'award',
    'prize': 'award',
    'honor': 'award',
    'chart': 'chart',
    'ranking': 'chart',
    'position': 'chart',
    'rank': 'chart',
    'hit': 'hit',
    'release date': 'release date',
    'release year': 'release year',
    'release month': 'release month',
    'release day': 'release day',
    'recorded date': 'recorded date',
    'written by': 'author',
    'composed by': 'composer',
    'produced by': 'producer'
}

relation_keys_sorted = sorted(relation_keywords.keys(), key=len, reverse=True)

def find_relation(phrase: str) -> Optional[str]:
    phrase_lc = phrase.lower()
    for rel in relation_keys_sorted:
        if rel in phrase_lc:
            return rel
    return None

def clean_entity(entity: str) -> str:
    entity = entity.strip()
    entity = re.sub(r'[.,;:!?]+$', '', entity)
    return entity

def make_subq(idgen: SubIdGen, question: str, entity_focus: Optional[str] = None, relation_hint: Optional[str] = None, hop_type: Optional[str] = None, references: List[str] = []) -> SubQuestion:
    return SubQuestion(sub_id=idgen.next(), question=question, entity_focus=entity_focus, relation_hint=relation_hint, hop_type=hop_type, references=references)

def extract_possessive_phrase(text: str) -> Optional[tuple]:
    m = re.match(r'^(?:the )?([\w\- ]+?) of (.+)$', text, re.I)
    if m:
        rel = m.group(1).strip()
        ent = m.group(2).strip()
        return rel, ent
    m2 = re.match(r'^([\w\- ]+?)\'s (.+)$', text, re.I)
    if m2:
        ent = m2.group(1).strip()
        rel = m2.group(2).strip()
        return rel, ent
    return None

def split_top_level_conjunctions(text: str) -> List[str]:
    parts = []
    depth = 0
    last = 0
    for i, c in enumerate(text):
        if c == '(': depth += 1
        elif c == ')':
            if depth > 0: depth -= 1
        elif depth == 0:
            if text[i:i+5].lower() == ' and ':
                parts.append(text[last:i].strip())
                last = i+5
            elif text[i:i+4].lower() == ' or ':
                parts.append(text[last:i].strip())
                last = i+4
    parts.append(text[last:].strip())
    return parts

def decompose(question: str) -> List[SubQuestion]:
    q = question.strip()
    idgen = SubIdGen()
    subs: List[SubQuestion] = []
    if not q.endswith('?'):
        q += '?'

    m = re.match(r'^(?:What|Who) is the ([^?]+) of (.+)\?$', q, re.I)
    if m:
        relation = m.group(1).strip()
        entity = clean_entity(m.group(2))
        hop_type = relation_keywords.get(relation.lower())
        sub1_id = idgen.next()
        subs.append(SubQuestion(sub_id=sub1_id, question=entity + '?', entity_focus=None, relation_hint=None, hop_type=None, references=[]))
        sub2_id = idgen.next()
        subs.append(SubQuestion(sub_id=sub2_id, question=f'What is the {relation} of #{sub1_id}?', entity_focus=relation, relation_hint=relation, hop_type=hop_type, references=[sub1_id]))
        return subs

    m = re.match(r'^Who is the person who (.+)\?$', q, re.I)
    if m:
        inner_q = m.group(1).strip()
        sub1_id = idgen.next()
        subs.append(SubQuestion(sub_id=sub1_id, question=inner_q + '?', entity_focus=None, relation_hint=None, hop_type=None, references=[]))
        sub2_id = idgen.next()
        subs.append(SubQuestion(sub_id=sub2_id, question=f'Who is the person who #{sub1_id}?', entity_focus=None, relation_hint=None, hop_type=None, references=[sub1_id]))
        return subs

    poss = extract_possessive_phrase(q[:-1])
    if poss:
        rel, ent = poss
        rel_clean = rel.lower()
        hop_type = relation_keywords.get(rel_clean)
        sub1_id = idgen.next()
        subs.append(SubQuestion(sub_id=sub1_id, question=ent + '?', entity_focus=None, relation_hint=None, hop_type=None, references=[]))
        sub2_id = idgen.next()
        subs.append(SubQuestion(sub_id=sub2_id, question=f'What is the {rel} of #{sub1_id}?', entity_focus=rel, relation_hint=rel, hop_type=hop_type, references=[sub1_id]))
        return subs

    parts = split_top_level_conjunctions(q[:-1])
    if len(parts) > 1:
        for part in parts:
            subq = make_subq(idgen, part + '?')
            subs.append(subq)
        return subs

    return []
