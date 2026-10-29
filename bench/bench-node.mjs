// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Verification-only HTTP/SSE instrument, derived from bench.py's request contract.
// Records raw monotonic nanoseconds. Rates and statistics belong in Wolfram.
import http from 'node:http';
import fs from 'node:fs';
import {StringDecoder} from 'node:string_decoder';
import {randomUUID} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

export async function oneRequest({port, model, prompt, maxTokens, timeoutMs=1800000}) {
    const row = {time_unit:'nanoseconds', clock:'process.hrtime.bigint',
        t_send_ns:null, t_first_ns:null, t_last_ns:null, chunks:0, text:'',
        usage:null, finish_reason:null, error:null, http_status:null, busy_retries:0};
    const body = JSON.stringify({model, messages:[{role:'user',content:prompt}],
        stream:true, stream_options:{include_usage:true}, temperature:0, top_p:1,
        max_tokens:maxTokens, seed:0, chat_template_kwargs:{enable_thinking:false}});
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const signal = controller.signal;
    const connect = () => new Promise((resolve,reject) => {
        const req = http.request({host:'127.0.0.1',port,path:'/v1/chat/completions',
            method:'POST',agent:false,signal,headers:{'Content-Type':'application/json',
            Accept:'text/event-stream'}}, resolve);
        req.on('error',reject);
        req.end(body);
    });
    try {
        row.t_send_ns = process.hrtime.bigint().toString();
        let response;
        while (true) {
            response = await connect();
            row.http_status = response.statusCode;
            if (![429,503].includes(response.statusCode)) break;
            response.resume();
            response.destroy();
            row.busy_retries++;
            await delay(50,undefined,{signal});
        }
        if (response.statusCode !== 200) {
            response.destroy();
            throw new Error('HTTP '+response.statusCode);
        }
        const decoder = new StringDecoder('utf8');
        let pending = '', done = false;
        try {
            stream: for await (const chunk of response) {
                pending += decoder.write(chunk);
                while (pending.includes('\n')) {
                    const boundary = pending.indexOf('\n');
                    const line = pending.slice(0,boundary).trim();
                    pending = pending.slice(boundary+1);
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (payload === '[DONE]') { done=true; break stream; }
                    let event;
                    try { event=JSON.parse(payload); }
                    catch { throw new Error('malformed SSE JSON'); }
                    if (event.error) throw new Error('server error event');
                    const now = process.hrtime.bigint().toString();
                    if (event.usage) row.usage=event.usage;
                    for (const choice of event.choices || []) {
                        const delta=choice.delta || {};
                        for(const key of ['content','reasoning_content'])
                            if(delta[key]!=null && typeof delta[key]!=='string')
                                throw new Error('invalid text delta');
                        const piece=(delta.content || '')+(delta.reasoning_content || '');
                        if (typeof piece !== 'string') throw new Error('invalid text delta');
                        if (piece) {
                            if (row.t_first_ns === null) row.t_first_ns=now;
                            row.t_last_ns=now;
                            row.chunks++;
                            row.text+=piece;
                        }
                        if (choice.finish_reason) row.finish_reason=choice.finish_reason;
                    }
                }
            }
        } finally { response.destroy(); }
        if (!done) throw new Error('stream ended without DONE');
        if (row.t_first_ns === null) throw new Error('no content received');
        if (!row.finish_reason) throw new Error('missing finish reason');
        if (!Number.isInteger(row.usage?.prompt_tokens) || !Number.isInteger(row.usage?.completion_tokens))
            throw new Error('missing token usage');
    } catch (error) { row.error=error.message; }
    finally { clearTimeout(timer); }
    return row;
}

async function main() {
    const args={};
    const flags=new Set(['port','model','arm','round','metric','prompts','select','concurrency',
        'max-tokens','warmup','reps','out','cell-run']);
    for (let i=2;i<process.argv.length;i++) {
        const name=process.argv[i].replace(/^--/,'');
        if(name==='nonce') { args.nonce=true; continue; }
        if(!flags.has(name) || process.argv[i+1]===undefined) throw new Error('invalid option '+name);
        args[name]=process.argv[++i];
    }
    for(const name of ['port','model','arm','round','metric','prompts','max-tokens','out'])
        if(!args[name]) throw new Error('missing option '+name);
    for(const [name,value] of Object.entries({concurrency:1,warmup:2,reps:3}))
        args[name]=Number(args[name] ?? value);
    args.port=Number(args.port); args['max-tokens']=Number(args['max-tokens']);
    for(const name of ['port','concurrency','warmup','reps','max-tokens'])
        if(!Number.isInteger(args[name]) || args[name]<0) throw new Error('invalid integer '+name);
    if(!args.port || !args.concurrency || !args.reps || !args['max-tokens']) throw new Error('zero size');
    let prompts=fs.readFileSync(args.prompts,'utf8').split('\n').filter(x=>x.trim()).map(JSON.parse);
    if(args.select) prompts=args.select.split(',').map(id=>{
        const p=prompts.find(p=>p.id===id); if(!p) throw new Error('missing prompt '+id); return p;
    });
    if(!prompts.length || prompts.some(p=>typeof p.id!=='string'||typeof p.text!=='string'))
        throw new Error('invalid prompt file');
    const output=fs.openSync(args.out,'wx'); // Preserve all prior runs; never silently append.
    let failed=false;
    try {
        for(let k=0;k<args.warmup+args.reps;k++) {
            const warmup=k<args.warmup, rep=warmup?k:k-args.warmup;
            const selected=Array.from({length:args.concurrency},(_,i)=>prompts[(k*args.concurrency+i)%prompts.length]);
            const batchId=randomUUID();
            const before=fs.readFileSync('/proc/loadavg','utf8').trim();
            const wallUtc=new Date().toISOString();
            const rows=await Promise.all(selected.map(p=>oneRequest({port:args.port,model:args.model,
                prompt:(args.nonce?'Request id '+randomUUID().replaceAll('-','')+'. Ignore this line.\n':'')+p.text,
                maxTokens:args['max-tokens']})));
            const after=fs.readFileSync('/proc/loadavg','utf8').trim();
            for(let i=0;i<rows.length;i++) {
                const row={...rows[i],arm:args.arm,round:args.round,metric:args.metric,rep,warmup,
                    concurrency:args.concurrency,stream_index:i,batch_id:batchId,
                    loadavg_start:before,loadavg_end:after,wall_utc:wallUtc,
                    max_tokens:args['max-tokens'],prompt_id:selected[i].id,cell_run:args['cell-run']||'',
                    server_cpus:process.env.H2H_SRV_CPUS||null,client_cpus:process.env.H2H_CLI_CPUS||null,
                    client:'node-sse-v1',node_version:process.version};
                fs.writeSync(output,JSON.stringify(row)+'\n');
                if(row.error) failed=true;
            }
        }
    } finally { fs.closeSync(output); }
    if(failed) process.exitCode=1;
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href)
    main().catch(error=>{console.error(error.message);process.exitCode=1;});
