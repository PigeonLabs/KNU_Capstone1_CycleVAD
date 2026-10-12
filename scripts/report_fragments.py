"""Preserve completed diagnostic sections when earlier reports are regenerated."""
def diagnostics_fragment(root):
    path=root/'docs/PAPER_DIAGNOSTICS_RESULTS.md'
    if not path.exists():return ''
    prefix=online_accuracy_fragment(root)+streaming_fragment(root)+calibration_fragment(root)
    return prefix+path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/').replace('](PAPER_STAGE2_DESIGN.md)','](docs/PAPER_STAGE2_DESIGN.md)').replace('](PAPER_CALIBRATION_RESULTS.md)','](docs/PAPER_CALIBRATION_RESULTS.md)')


def calibration_fragment(root):
    path=root/'docs/PAPER_CALIBRATION_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/').replace('](PAPER_NEXT_EXPERIMENTS_20261009.md)','](docs/PAPER_NEXT_EXPERIMENTS_20261009.md)').replace('](R01_AUDIT_GUIDE.md)','](docs/R01_AUDIT_GUIDE.md)')


def streaming_fragment(root):
    path=root/'docs/STREAMING_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](figures/','](docs/figures/')


def online_accuracy_fragment(root):
    path=root/'docs/ONLINE_ACCURACY_RESULTS.md'
    if not path.exists():return ''
    return path.read_text().replace('](../results/','](results/').replace('](../configs/','](configs/').replace('](../scripts/','](scripts/').replace('](figures/','](docs/figures/')


def sync_readme_results(root):
    """Keep every detailed result inline, ordered independently of report arrival."""
    import re
    archive=(root/'EXPERIMENTS.md').read_text()
    def sections(text,level):
        prefix=[];parts=[];current=prefix;fenced=False
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith('```'):fenced=not fenced
            if not fenced and line.startswith('#'*level+' '):
                current=[];parts.append((line[level+1:].strip(),current))
            else:current.append(line)
        return ''.join(prefix),[(title,''.join(body)) for title,body in parts]
    def shift(text,amount):
        lines=[];fenced=False
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith('```'):fenced=not fenced
            if not fenced and line.startswith('#'):line='#'*amount+line
            lines.append(line)
        return ''.join(lines)
    _,groups=sections(archive,2);records=[]
    for title,body in groups:
        if title in ['실험 상태','재현','한계']:continue
        grouped=title.startswith(('후속 실험 E6','위치·진행 검증','논문 통합 실험','최소 구성과 오경보 진단','보정과 경보 결합'))
        if grouped:
            prefix,children=sections(body,3)
            for child,content in children:
                if re.match(r'E\d',child):
                    records.append({'title':child,'body':prefix+content,'level':3});prefix=''
                else:
                    # Keep historical plans and reproduction notes with their results.
                    if child=='다음 실험 계획':child='당시 후속 계획과 결과 연결'
                    records[-1]['body']+='\n#### '+child+'\n\n'+content
        else:records.append({'title':title,'body':body,'level':2})
    def order(record):
        title=record['title']
        if title=='방법과 평가 규칙':return (-2,'')
        if title.startswith('초기 실험'):return (-1,'')
        if title=='실행 비용':return (5,'Z')
        match=re.search(r'E(\d+)([A-Z0-9/]*)',title)
        if not match:raise ValueError('Unclassified result section: '+title)
        suffix=match[2];suffix={'B0':'B0','B/C':'B1'}.get(suffix,suffix)
        return (int(match[1]),suffix)
    records.sort(key=order)
    seen={};blocks=[];toc=[]
    for record in records:
        number,suffix=order(record)
        key={-2:'evaluation-detail',-1:'early-notes'}.get(number,'e'+str(number)+suffix.lower().replace('/','-'))
        seen[key]=seen.get(key,0)+1
        anchor='result-'+key+('' if seen[key]==1 else '-'+str(seen[key]))
        title=record['title']
        if number==-2:label='평가 규칙'
        elif number==-1:label='초기 요약'
        elif suffix=='Z':label='초기 실행 비용'
        else:
            label=re.search(r'E\d+[A-Z0-9/]*',title)[0]
            if number==8 and not suffix:label+=' (seed 42)' if seen[key]==1 else ' (seed 43·44)'
        toc.append(f'[{label}](#{anchor})')
        blocks.append(f'<a id="{anchor}"></a>\n\n### {title}\n\n'+shift(record['body'],3-record['level']).strip()+'\n')
    begin='<!-- BEGIN EXPERIMENT RESULTS -->';end='<!-- END EXPERIMENT RESULTS -->'
    path=root/'README.md';readme=path.read_text()
    if readme.count(begin)!=1 or readme.count(end)!=1:raise ValueError('README must contain exactly one result region')
    body='\n\n'+' · '.join(toc)+'\n\n'+'\n'.join(blocks)+'\n'
    path.write_text(readme[:readme.index(begin)+len(begin)]+body+readme[readme.index(end):])
