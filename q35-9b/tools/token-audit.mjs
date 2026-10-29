// Verification-only: pinned CPU llama-tokenize against the recorded HF oracle.
// Counts/positions are retained; rates and statistics are calculated in Wolfram.
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const jsonl=p=>fs.readFileSync(p,'utf8').split('\n').filter(x=>x.trim()).map(JSON.parse);

export function compareTokens(a,b) {
    for(const ids of [a,b])
        if(!Array.isArray(ids)||ids.some(x=>!Number.isInteger(x)||x<0)) throw new Error('invalid token IDs');
    let prefix=0;
    while(prefix<a.length && prefix<b.length && a[prefix]===b[prefix]) prefix++;
    const identical=prefix===a.length && prefix===b.length;
    return {length_a:a.length,length_b:b.length,common_prefix:prefix,identical,
        first_div:identical?null:prefix,token_a:a[prefix]??null,token_b:b[prefix]??null};
}
export function uniqueRows(rows) {
    const map=new Map();
    for(const row of rows) {
        if(typeof row.prompt_id!=='string'||map.has(row.prompt_id)) throw new Error('missing or duplicate prompt ID');
        map.set(row.prompt_id,row);
    }
    return map;
}
export function findObservation(records,prompt,comparison) {
    if(comparison.identical || comparison.token_a===null || comparison.token_b===null) return null;
    return records.find(r=>r.prompt_id===prompt && r.first_div===comparison.first_div &&
        r.oracle_tok===comparison.token_a && r.arm_tok===comparison.token_b &&
        r.oracle_len===comparison.length_a && Number.isFinite(r.logit_gap)) || null;
}
function main() {
    const [tokenizer,gguf,out,oraclePath,tiesCSV,...armArgs]=process.argv.slice(2);
    if(!tokenizer||!gguf||!out||!oraclePath||!tiesCSV||!armArgs.length) throw new Error('missing arguments');
    if(fs.existsSync(out)) throw new Error('output already exists');
    const prompts=jsonl(new URL('../../prompts/decode.jsonl',import.meta.url));
    const oracle=uniqueRows(jsonl(oraclePath));
    if(prompts.length!==oracle.size || prompts.some(p=>!oracle.has(p.id))) throw new Error('oracle prompt coverage mismatch');
    const log=fs.openSync(out+'.tokenizer.log','wx');
    const tokens=text=>{
        if(typeof text!=='string') throw new Error('missing output text');
        const run=spawnSync(tokenizer,['-m',gguf,'--stdin','--ids','--no-bos','--no-parse-special','--no-escape'],
            {input:text,encoding:'utf8',maxBuffer:16777216,timeout:60000});
        fs.writeSync(log,run.stderr||'');
        if(run.status!==0) throw new Error('tokenizer failed: '+run.status);
        const ids=JSON.parse(run.stdout);
        compareTokens(ids,ids); // Validate IDs even when no comparison follows.
        return ids;
    };
    try {
        const report={kind:'token_audit_not_final_model_qualification',
            tokenizer:{path:tokenizer,sha256:hash(tokenizer),source:'7aca0d2ea8d92b56911233b822db01f9fbb1ae50',backend:'CPU-only'},
            oracle:{path:oraclePath,sha256:hash(oraclePath)},tieSources:[],oracleRoundtrip:[],arms:{},pairs:{}};
        const observations=[];
        for(const path of tiesCSV.split(',')) {
            report.tieSources.push({path,sha256:hash(path)});observations.push(...jsonl(path));
        }
        for(const [id,row] of oracle) report.oracleRoundtrip.push({prompt_id:id,...compareTokens(row.ids,tokens(row.text))});
        report.tokenizerValidatedAgainstRecordedHF=report.oracleRoundtrip.every(r=>r.identical);
        const allTokens={};
        for(const arg of armArgs) {
            const equal=arg.indexOf('=');if(equal<1) throw new Error('invalid arm argument');
            const arm=arg.slice(0,equal),path=arg.slice(equal+1);
            if(allTokens[arm]) throw new Error('duplicate arm');
            const input=uniqueRows(jsonl(path));
            if(input.size!==oracle.size || [...oracle.keys()].some(id=>!input.has(id))) throw new Error('arm prompt coverage mismatch');
            allTokens[arm]={};
            report.arms[arm]={path,sha256:hash(path),rows:[]};
            for(const [id,reference] of oracle) {
                const row=input.get(id);
                if(row.error!==null || row.http_status!==200 || row.warmup || row.metric!=='agree') throw new Error('invalid correctness row');
                const ids=tokens(row.text);allTokens[arm][id]=ids;
                const comparison=compareTokens(reference.ids,ids);
                const observation=report.tokenizerValidatedAgainstRecordedHF?findObservation(observations,id,comparison):null;
                report.arms[arm].rows.push({prompt_id:id,...comparison,
                    recordedUsage:row.usage?.completion_tokens??null,
                    priorObservation:observation,
                    observationStatus:comparison.identical?'IDENTICAL':observation?'EXACT_PRIOR_TUPLE':'NEEDS_ORACLE_OBSERVATION'});
            }
        }
        const names=Object.keys(allTokens);
        for(let a=0;a<names.length;a++) for(let b=a+1;b<names.length;b++)
            report.pairs[names[a]+'-'+names[b]]=[...oracle.keys()].map(id=>({prompt_id:id,
                ...compareTokens(allTokens[names[a]][id],allTokens[names[b]][id])}));
        report.notes=['Recorded prior logit observations are reused only for exact prompt/position/token tuples against the same frozen HF oracle.',
            'Matching prior scores are observations, not proof of all-model numerical equivalence.',
            'Uncovered divergences remain unverified. No speed or model qualification verdict follows from request success alone.'];
        fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
        console.log(JSON.stringify({tokenizerValidatedAgainstRecordedHF:report.tokenizerValidatedAgainstRecordedHF,
            arms:Object.fromEntries(Object.entries(report.arms).map(([a,v])=>[a,v.rows.map(r=>({id:r.prompt_id,status:r.observationStatus,first:r.first_div,gap:r.priorObservation?.logit_gap??null}))]))}));
        if(!report.tokenizerValidatedAgainstRecordedHF) process.exitCode=1;
    } finally { fs.closeSync(log); }
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href)
    try { main(); } catch(error) {console.error(error.message);process.exitCode=1;}
