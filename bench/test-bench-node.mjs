// Verification-only synthetic HTTP controls. No model or GPU required.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {oneRequest} from './bench-node.mjs';

const event=value=>'data: '+JSON.stringify(value)+'\r\n\r\n';
const content=text=>event({choices:[{delta:{content:text},finish_reason:null}]});
const finish=event({choices:[{delta:{},finish_reason:'length'}],usage:{prompt_tokens:5,completion_tokens:2}});
const body=content('héllo')+finish+'data: [DONE]\r\n\r\n';
async function server(handler,run) {
    const s=http.createServer((req,res)=>{
        let input='';req.on('data',chunk=>input+=chunk);
        req.on('end',()=>Promise.resolve(handler(req,res,JSON.parse(input))).catch(e=>res.destroy(e)));
    });
    await new Promise(resolve=>s.listen(0,'127.0.0.1',resolve));
    try { return await run(s.address().port); }
    finally { s.closeAllConnections(); await new Promise(resolve=>s.close(resolve)); }
}
const call=(port,extra={})=>oneRequest({port,model:'fixture',prompt:'public fixture',maxTokens:2,...extra});

test('request contract, fragmented UTF-8/SSE, reasoning, usage and raw clocks',async()=>{
    await server((req,res,b)=>{
        assert.equal(req.url,'/v1/chat/completions');assert.equal(req.method,'POST');
        assert.deepEqual(b,{model:'fixture',messages:[{role:'user',content:'public fixture'}],stream:true,
            stream_options:{include_usage:true},temperature:0,top_p:1,max_tokens:2,seed:0,
            chat_template_kwargs:{enable_thinking:false}});
        res.writeHead(200,{'Content-Type':'text/event-stream'});
        const bytes=Buffer.from(': ignored\r\n'+event({choices:[{delta:{reasoning_content:'réason '}}]})+body);
        for(const byte of bytes) res.write(Buffer.from([byte]));
        res.end();
    },async port=>{
        const row=await call(port);assert.equal(row.error,null);assert.equal(row.text,'réason héllo');
        assert.equal(row.chunks,2);assert.equal(row.finish_reason,'length');
        assert.deepEqual(row.usage,{prompt_tokens:5,completion_tokens:2});
        for(const key of ['t_send_ns','t_first_ns','t_last_ns']) assert.match(row[key],/^\d+$/);
    });
});
test('busy retries retain one request start and eventually succeed',async()=>{
    let requests=0;
    await server((req,res)=>{requests++;if(requests===1){res.writeHead(503);res.end('busy');}
        else{res.writeHead(200);res.end(body);}},async port=>{
        const row=await call(port);assert.equal(row.error,null);assert.equal(row.busy_retries,1);
        assert.equal(row.http_status,200);assert.equal(requests,2);
    });
});
for(const [name,payload,message] of [
    ['malformed','data: {bad}\n\n','malformed SSE JSON'],
    ['truncated',content('partial')+finish,'stream ended without DONE'],
    ['missing usage',content('x')+event({choices:[{finish_reason:'stop'}]})+'data: [DONE]\n','missing token usage'],
    ['no content',finish+'data: [DONE]\n','no content received'],
    ['server error',event({error:{message:'fixture'}}),'server error event'],
    ['bad delta',event({choices:[{delta:{content:{bad:true}}}]}),'invalid text delta'],
]) test(name+' is recorded as a failed request',async()=>{
    await server((req,res)=>{res.writeHead(200);res.end(payload);},async port=>{
        const row=await call(port);assert.equal(row.error,message);
    });
});
test('non-success HTTP and overall retry timeout refuse success',async()=>{
    await server((req,res)=>{res.writeHead(400);res.end('fixture');},async port=>{
        assert.equal((await call(port)).error,'HTTP 400');
    });
    await server((req,res)=>{res.writeHead(429);res.end('busy');},async port=>{
        const row=await call(port,{timeoutMs:50});assert(row.error);assert.equal(row.t_first_ns,null);
    });
});
const client=fileURLToPath(new URL('./bench-node.mjs',import.meta.url));
const cli=args=>new Promise(resolve=>{
    const child=spawn(process.execPath,[client,...args],{stdio:['ignore','pipe','pipe']});
    let stderr='';child.stderr.on('data',x=>stderr+=x);child.on('close',status=>resolve({status,stderr}));
});
test('CLI writes all concurrent rows, nonce, warmup flags, and refuses overwrite',async()=>{
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'bench-node-control-'));
    try {
        const prompts=path.join(root,'prompts.jsonl'),out=path.join(root,'rows.jsonl');
        fs.writeFileSync(prompts,'{"id":"p0","text":"zero"}\n{"id":"p1","text":"one"}\n');
        const seen=[];
        await server((req,res,b)=>{seen.push(b.messages[0].content);res.writeHead(200);res.end(body);},async port=>{
            const args=['--port',String(port),'--model','fixture','--arm','CONTROL','--round','R0',
                '--metric','decode','--prompts',prompts,'--max-tokens','2','--concurrency','2',
                '--warmup','1','--reps','1','--nonce','--out',out];
            assert.equal((await cli(args)).status,0);
            const rows=fs.readFileSync(out,'utf8').trim().split('\n').map(JSON.parse);
            assert.equal(rows.length,4);assert.deepEqual(rows.map(r=>r.warmup),[true,true,false,false]);
            assert.deepEqual(rows.map(r=>r.prompt_id),['p0','p1','p0','p1']);
            assert(rows.every(r=>r.error===null&&r.client==='node-sse-v1'&&r.concurrency===2));
            assert(seen.every(t=>/^Request id [a-f0-9]+\. Ignore this line\.\n/.test(t)));
            assert.equal(new Set(seen).size,4);
            assert.notEqual((await cli(args)).status,0);assert.equal(seen.length,4);
        });
    } finally { fs.rmSync(root,{recursive:true,force:true}); }
});
