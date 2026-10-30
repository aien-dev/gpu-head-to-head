import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {reorder,widenBf16,mapping} from './ordered-weights.mjs';
import {ggufCatalog,safetensorsCatalog} from './weight-files.mjs';
const c={linear_num_key_heads:2,linear_num_value_heads:6,num_hidden_layers:1,linear_value_head_dim:2,linear_key_head_dim:1};
const words=xs=>{const b=Buffer.alloc(xs.length*2);xs.forEach((x,i)=>b.writeUInt16LE(x,i*2));return b;};
const unpack=b=>Array.from({length:b.length/2},(_,i)=>b.readUInt16LE(i*2));
// Golden addresses from the recorded Wolfram reshape/transpose query, not from this implementation.
const golden=[0,1,6,7,2,3,8,9,4,5,10,11];
test('row and column permutations match independent Wolfram address fixtures',()=>{
    const data=words(Array.from({length:12},(_,i)=>i));
    assert.deepEqual(unpack(reorder(data,[12],2,c,{axis:0,hd:2})),golden);
    const two=words(Array.from({length:24},(_,i)=>i));
    assert.deepEqual(unpack(reorder(two,[2,12],2,c,{axis:1,hd:2})),
        [0,1,6,7,2,3,8,9,4,5,10,11,12,13,18,19,14,15,20,21,16,17,22,23]);
});
test('Q/K prefix stays fixed while only V rows move',()=>{
    const data=words(Array.from({length:16},(_,i)=>i));
    assert.deepEqual(unpack(reorder(data,[16],2,c,{axis:0,hd:2,prefix:4})),
        [0,1,2,3,4,5,10,11,6,7,12,13,8,9,14,15]);
});
test('same-multiset wrong order is detected rather than accepted',()=>{
    const src=words(Array.from({length:12},(_,i)=>i)),good=reorder(src,[12],2,c,{axis:0,hd:2});
    assert.deepEqual(unpack(src).sort((a,b)=>a-b),unpack(good).sort((a,b)=>a-b));
    assert(!src.equals(good));
    assert.throws(()=>reorder(src,[12],2,{...c,linear_num_key_heads:4},{axis:0,hd:2}));
    assert.throws(()=>reorder(src,[11],2,c,{axis:0,hd:2}));
});
test('BF16 widening copies bits and malformed bytes fail',()=>{
    assert.deepEqual([...widenBf16(Buffer.from([0x80,0x3f,0,0x80]))],[0,0,0x80,0x3f,0,0,0,0x80]);
    assert.throws(()=>widenBf16(Buffer.from([1])));
});
test('names and norm roles are explicit, with no unknown text-tensor fallback',()=>{
    assert.equal(mapping('model.language_model.layers.0.post_attention_layernorm.weight',c).name,'blk.0.post_attention_norm.weight');
    assert.equal(mapping('model.language_model.layers.0.linear_attn.in_proj_qkv.weight',c).prefix,4);
    assert.equal(mapping('mtp.fc.weight',c),null);
    assert.throws(()=>mapping('model.language_model.layers.0.unknown.weight',c));
});
const u32=n=>{const b=Buffer.alloc(4);b.writeUInt32LE(n);return b;};
const u64=n=>{const b=Buffer.alloc(8);b.writeBigUInt64LE(BigInt(n));return b;};
const str=s=>Buffer.concat([u64(Buffer.byteLength(s)),Buffer.from(s)]);
test('binary readers accept exact fixtures and refuse truncation or bad types',()=>{
    const root=fs.mkdtempSync(path.join(os.tmpdir(),'ordered-weight-control-'));
    try{
        const file=path.join(root,'model.gguf');
        const header=Buffer.concat([Buffer.from('GGUF'),u32(3),u64(1),u64(0),str('fixture'),u32(1),u64(12),u32(30),u64(0)]);
        const pad=Buffer.alloc((32-header.length%32)%32);
        const bytes=Buffer.concat([header,pad,words(Array.from({length:12},(_,i)=>i))]);
        fs.writeFileSync(file,bytes);
        const g=ggufCatalog(file);assert.equal(g.tensors.get('fixture').bytes,24);g.close();
        fs.writeFileSync(file,bytes.subarray(0,bytes.length-1));assert.throws(()=>ggufCatalog(file));
        fs.writeFileSync(file,Buffer.from('GGUF'));assert.throws(()=>ggufCatalog(file));
        const safepath=path.join(root,'fixture.safetensors');
        const h=Buffer.from(JSON.stringify({fixture:{dtype:'BF16',shape:[12],data_offsets:[0,24]}}));
        fs.writeFileSync(safepath,Buffer.concat([u64(h.length),h,words(Array.from({length:12},(_,i)=>i))]));
        const s=safetensorsCatalog(root);assert.equal(s.tensors.size,1);s.close();
        fs.writeFileSync(safepath,Buffer.concat([u64(h.length),h]));assert.throws(()=>safetensorsCatalog(root));
    }finally{fs.rmSync(root,{recursive:true,force:true});}
});
