// Verification-only byte/address glue. Numeric transforms call seeded Omega.
// Converter reference: llama.cpp 7aca0d2e conversion/qwen.py, lines 404 and 463-655.
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {readAt,ggufCatalog,safetensorsCatalog} from './weight-files.mjs';
const sha=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');

export function mapping(name,c) {
    if(name==='lm_head.weight')return {name:'output.weight'};
    if(name.startsWith('mtp.')||name.startsWith('model.visual.'))return null;
    const prefix='model.language_model.';
    if(!name.startsWith(prefix))throw Error('unknown HF namespace: '+name);
    name=name.slice(prefix.length);
    if(name==='embed_tokens.weight')return {name:'token_embd.weight'};
    if(name==='norm.weight')return {name:'output_norm.weight',numeric:'plus1'};
    const m=/^layers\.(\d+)\.(.+)$/.exec(name);if(!m||Number(m[1])>=c.num_hidden_layers)throw Error('unknown text tensor');
    const base='blk.'+m[1]+'.',tail=m[2];
    const simple={'mlp.down_proj.weight':'ffn_down.weight','mlp.up_proj.weight':'ffn_up.weight',
        'mlp.gate_proj.weight':'ffn_gate.weight','self_attn.q_proj.weight':'attn_q.weight',
        'self_attn.k_proj.weight':'attn_k.weight','self_attn.v_proj.weight':'attn_v.weight',
        'self_attn.o_proj.weight':'attn_output.weight','linear_attn.norm.weight':'ssm_norm.weight'};
    if(simple[tail])return {name:base+simple[tail]};
    const norms={'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'post_attention_norm.weight',
        'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight'};
    if(norms[tail])return {name:base+norms[tail],numeric:'plus1'};
    const qk=2*c.linear_num_key_heads*c.linear_key_head_dim;
    const linear={
        'in_proj_qkv.weight':{target:'attn_qkv.weight',axis:0,prefix:qk,hd:c.linear_value_head_dim},
        'in_proj_z.weight':{target:'attn_gate.weight',axis:0,hd:c.linear_value_head_dim},
        'in_proj_a.weight':{target:'ssm_alpha.weight',axis:0,hd:1},
        'in_proj_b.weight':{target:'ssm_beta.weight',axis:0,hd:1},
        'out_proj.weight':{target:'ssm_out.weight',axis:1,hd:c.linear_value_head_dim},
        'conv1d.weight':{target:'ssm_conv1d.weight',axis:0,prefix:qk,hd:c.linear_value_head_dim,squeeze:true},
        'dt_bias':{target:'ssm_dt.bias',axis:0,hd:1},
        'A_log':{target:'ssm_a',axis:0,hd:1,numeric:'negexp'}
    };
    const v=tail.startsWith('linear_attn.')?linear[tail.slice('linear_attn.'.length)]:null;
    if(!v)throw Error('unmapped text tensor: '+name);
    return {...v,name:base+v.target};
}
export function reorder(data,shape,bytes,c,plan) {
    if(plan.axis===undefined)return Buffer.from(data);
    const nk=c.linear_num_key_heads,nv=c.linear_num_value_heads,per=nv/nk,hd=plan.hd,prefix=plan.prefix||0;
    if(!Number.isInteger(per)||per<=0||!Number.isInteger(hd)||hd<=0)throw Error('invalid head mapping');
    if(shape[plan.axis]!==prefix+nv*hd)throw Error('head axis does not match tensor');
    const out=Buffer.alloc(data.length);
    if(plan.axis===0){
        const stride=shape.slice(1).reduce((a,b)=>a*b,1)*bytes;
        data.copy(out,0,0,prefix*stride);
        for(let v=0;v<per;v++)for(let k=0;k<nk;k++){
            const src=(prefix+(k*per+v)*hd)*stride,dst=(prefix+(v*nk+k)*hd)*stride;
            data.copy(out,dst,src,src+hd*stride);
        }
    }else{
        if(plan.axis!==1||shape.length!==2||prefix)throw Error('unsupported permutation axis');
        const rowBytes=shape[1]*bytes;
        for(let row=0;row<shape[0];row++)for(let v=0;v<per;v++)for(let k=0;k<nk;k++){
            const src=row*rowBytes+(k*per+v)*hd*bytes,dst=row*rowBytes+(v*nk+k)*hd*bytes;
            data.copy(out,dst,src,src+hd*bytes);
        }
    }
    return out;
}
export function widenBf16(data){
    if(data.length%2)throw Error('odd BF16 byte count');
    const out=Buffer.alloc(data.length*2);
    for(let i=0;i<data.length;i+=2){out[i*2+2]=data[i];out[i*2+3]=data[i+1];}
    return out;
}
function difference(a,b){
    if(a.length!==b.length)return {kind:'length',expected:a.length,actual:b.length};
    if(a.equals(b))return null;
    for(let i=0;i<a.length;i++)if(a[i]!==b[i])return {kind:'byte',offset:i,expected:a[i],actual:b[i]};
    throw Error('inconsistent buffer comparison');
}
function compareDirect(h,g){
    if(h.bytes!==g.bytes)return {kind:'length'};
    for(let at=0;at<h.bytes;at+=1048576){const n=Math.min(1048576,h.bytes-at);
        const d=difference(readAt(h.fd,h.offset+at,n),readAt(g.fd,g.offset+at,n));
        if(d)return {...d,chunkOffset:at};}
    return null;
}
function main(){
    const [hfDir,ggufFile,numeric,out,...options]=process.argv.slice(2);
    if(!hfDir||!ggufFile||!numeric||!out||fs.existsSync(out))throw Error('missing arguments or existing output');
    const mutant=options.includes('--mutant-order'),only=options.find(x=>x.startsWith('--only='))?.slice(7);
    const expHelper=options.find(x=>x.startsWith('--negexp-helper='))?.slice('--negexp-helper='.length);
    if(options.some(x=>x!=='--mutant-order'&&!x.startsWith('--only=')&&!x.startsWith('--negexp-helper=')))throw Error('unknown option');
    const configFile=hfDir+'/config.json',c=JSON.parse(fs.readFileSync(configFile)).text_config;
    for(const k of ['num_hidden_layers','linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim'])
        if(!Number.isSafeInteger(c[k])||c[k]<=0)throw Error('invalid model config');
    const hf=safetensorsCatalog(hfDir),g=ggufCatalog(ggufFile);
    try {
        const plans=[],covered=new Set(),excludedHF=[];
        for(const t of hf.tensors.values()){
            const p=mapping(t.name,c);if(!p){excludedHF.push(t.name);continue;}
            const target=g.tensors.get(p.name);if(!target||covered.has(p.name))throw Error('missing/duplicate named GGUF tensor: '+p.name);
            const shape=p.squeeze?t.shape.filter(x=>x!==1):t.shape;
            if(JSON.stringify(shape)!==JSON.stringify(target.shape))throw Error('shape mismatch: '+p.name);
            covered.add(p.name);plans.push({h:t,g:target,plan:p,shape});
        }
        const excludedGGUF=[];
        for(const name of g.tensors.keys())if(!covered.has(name)){
            const m=/^blk\.(\d+)\./.exec(name);
            if(name.startsWith('nextn.')||(m&&Number(m[1])===c.num_hidden_layers))excludedGGUF.push(name);
            else throw Error('unaccounted GGUF tensor: '+name);
        }
        const report={kind:'ordered_weight_verification',source:'llama.cpp7aca0d2e conversion/qwen.py',
            numeric:expHelper?'seeded Omega plus-one; pinned ATen CPU exp oracle':'seeded Omega bin/sexp units via I/O-only glue',
            numericBinarySha256:sha(numeric),expHelper:expHelper?{path:expHelper,sha256:sha(expHelper)}:null,
            configSha256:sha(configFile),mutantOrder:mutant,only:only||null,
            catalogue:{hf:hf.tensors.size,gguf:g.tensors.size,textPlans:plans.length,excludedHF,excludedGGUF},rows:[],expEvidence:[]};
        const selected=only?plans.filter(p=>p.g.name===only):plans;if(!selected.length)throw Error('empty tensor selection');
        for(const {h,g,plan,shape}of selected){
            let mismatch;
            if(plan.axis===undefined&&!plan.numeric&&h.dtype===g.dtype)mismatch=compareDirect(h,g);
            else{
                let data=readAt(h.fd,h.offset,h.bytes);
                if(h.dtype!==g.dtype){if(h.dtype!=='BF16'||g.dtype!=='F32')throw Error('unhandled numeric format');data=widenBf16(data);}
                const input=reorder(data,shape,g.dtype==='BF16'?2:4,c,mutant?{...plan,axis:undefined}:plan);
                let expected=input;
                if(plan.numeric){
                    if(g.dtype!=='F32')throw Error('numeric transform requires F32');
                    const helper=plan.numeric==='negexp'&&expHelper?expHelper:numeric;
                    const result=spawnSync(helper,[plan.numeric],{input,timeout:60000,maxBuffer:1048576});
                    if(result.status!==0||result.stdout.length!==input.length)throw Error('numeric transform failed: '+result.status);
                    expected=result.stdout;
                }
                const actual=readAt(g.fd,g.offset,g.bytes);mismatch=difference(expected,actual);
                if(plan.numeric==='negexp')report.expEvidence.push({tensor:g.name,
                    inputBits:Array.from({length:input.length/4},(_,i)=>input.readUInt32LE(i*4)),
                    referenceBackend:expHelper?'pinned ATen CPU':'seeded Omega sexp',
                    expectedBits:Array.from({length:expected.length/4},(_,i)=>expected.readUInt32LE(i*4)),
                    actualBits:Array.from({length:actual.length/4},(_,i)=>actual.readUInt32LE(i*4))});
            }
            report.rows.push({hf:h.name,gguf:g.name,shape,hfType:h.dtype,ggufType:g.dtype,
                permutationAxis:plan.axis??null,numeric:plan.numeric||null,status:mismatch?'MISMATCH':'EXACT',mismatch});
        }
        report.status=report.rows.some(r=>r.status!=='EXACT')?'MISMATCH':only?'PARTIAL_SELECTION_EXACT':'ALL_NAMED_TEXT_TENSORS_EXACT';
        fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
        console.log(JSON.stringify({status:report.status,checked:report.rows.length,
            mismatches:report.rows.filter(r=>r.status!=='EXACT').map(r=>({name:r.gguf,transform:r.numeric,axis:r.permutationAxis,mismatch:r.mismatch}))}));
        if(report.status==='MISMATCH')process.exitCode=1;
    }finally{hf.close();g.close();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)
    try{main();}catch(e){console.error(e.message);process.exitCode=1;}
