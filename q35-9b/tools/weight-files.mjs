// Verification-only, read-only binary catalogues. Layouts from llama.cpp 7aca0d2e
// gguf-py/gguf/{constants,gguf_reader}.py and safetensors' JSON header.
import fs from 'node:fs';
import path from 'node:path';
export function readAt(fd,pos,size) {
    if(!Number.isSafeInteger(pos)||!Number.isSafeInteger(size)||pos<0||size<0) throw Error('invalid byte range');
    const data=Buffer.alloc(size);let used=0;
    while(used<size) {const n=fs.readSync(fd,data,used,size-used,pos+used);if(!n)throw Error('truncated file');used+=n;}
    return data;
}
class Reader {
    constructor(file){this.fd=fs.openSync(file,'r');this.size=fs.fstatSync(this.fd).size;this.pos=0;this.cache=Buffer.alloc(0);this.base=0;}
    skip(n){if(!Number.isSafeInteger(n)||n<0||this.pos+n>this.size)throw Error('invalid skip');this.pos+=n;}
    bytes(n){
        if(!Number.isSafeInteger(n)||n<0||n>65536||this.pos+n>this.size)throw Error('invalid header size');
        if(this.pos<this.base||this.pos+n>this.base+this.cache.length){this.base=this.pos;this.cache=readAt(this.fd,this.pos,Math.min(1048576,this.size-this.pos));}
        const b=this.cache.subarray(this.pos-this.base,this.pos-this.base+n);this.pos+=n;return b;
    }
    u32(){return this.bytes(4).readUInt32LE();}
    u64(){const n=this.bytes(8).readBigUInt64LE();if(n>BigInt(Number.MAX_SAFE_INTEGER))throw Error('oversize integer');return Number(n);}
    string(keep=true){const n=this.u64();if(!keep){this.skip(n);return null;}return this.bytes(n).toString('utf8');}
    value(type,keep=false){
        const widths={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8};
        if(type===8)return this.string(keep);
        if(type===9){const sub=this.u32(),n=this.u64();if(widths[sub]){this.skip(n*widths[sub]);return null;}
            if(sub!==8)throw Error('unsupported metadata array');for(let i=0;i<n;i++)this.string(false);return null;}
        if(!widths[type])throw Error('unknown metadata type');
        const b=this.bytes(widths[type]);if(!keep)return null;
        if(type!==4)throw Error('alignment must be uint32');return b.readUInt32LE();
    }
}
export function ggufCatalog(file){
    const r=new Reader(file);
    try {
        if(r.bytes(4).toString()!=='GGUF'||r.u32()!==3)throw Error('expected little-endian GGUF v3');
        const count=r.u64(),kv=r.u64();let alignment=32;
        for(let i=0;i<kv;i++){const key=r.string(),type=r.u32();const v=r.value(type,key==='general.alignment');if(key==='general.alignment')alignment=v;}
        if(!Number.isSafeInteger(alignment)||alignment<=0||(alignment&(alignment-1)))throw Error('bad alignment');
        const tensors=new Map();
        for(let i=0;i<count;i++){
            const name=r.string(),rank=r.u32();if(!rank||rank>4||tensors.has(name))throw Error('bad tensor descriptor');
            const dims=Array.from({length:rank},()=>r.u64()),type=r.u32(),relative=r.u64();
            if(dims.some(x=>x<=0)||![0,30].includes(type)||relative%alignment)throw Error('unsupported tensor shape/type/offset');
            const bytes=dims.reduce((a,b)=>a*b,1)*(type===30?2:4);
            if(!Number.isSafeInteger(bytes))throw Error('tensor too large');
            tensors.set(name,{name,fd:r.fd,file,shape:dims.reverse(),dtype:type===30?'BF16':'F32',bytes,relative});
        }
        const padding=r.pos%alignment;const base=padding?r.pos+alignment-padding:r.pos;
        let end=base;
        for(const t of [...tensors.values()].sort((a,b)=>a.relative-b.relative)){
            t.offset=base+t.relative;if(t.offset<end||t.offset+t.bytes>r.size)throw Error('overlap or truncated tensor');end=t.offset+t.bytes;
        }
        return {tensors,alignment,dataOffset:base,close:()=>fs.closeSync(r.fd)};
    }catch(e){fs.closeSync(r.fd);throw e;}
}
export function safetensorsCatalog(directory){
    const tensors=new Map(),fds=[];
    try {
        for(const file of fs.readdirSync(directory).filter(f=>f.endsWith('.safetensors')).sort()){
            const full=path.join(directory,file),fd=fs.openSync(full,'r');fds.push(fd);
            const size=fs.fstatSync(fd).size,nBig=readAt(fd,0,8).readBigUInt64LE();
            if(nBig>16777216n||nBig>BigInt(size-8))throw Error('invalid safetensors header');
            const n=Number(nBig),header=JSON.parse(readAt(fd,8,n).toString('utf8'));let end=8+n;
            const entries=[];
            for(const [name,v]of Object.entries(header)){
                if(name==='__metadata__')continue;
                if(tensors.has(name)||!['BF16','F32'].includes(v.dtype)||!Array.isArray(v.shape)||v.shape.some(d=>!Number.isSafeInteger(d)||d<=0))throw Error('bad safetensors descriptor');
                if(!Array.isArray(v.data_offsets)||v.data_offsets.length!==2||v.data_offsets.some(d=>!Number.isSafeInteger(d)||d<0))throw Error('bad data offsets');
                const bytes=v.shape.reduce((a,b)=>a*b,1)*(v.dtype==='BF16'?2:4);
                const [a,b]=v.data_offsets;if(!Number.isSafeInteger(bytes)||b-a!==bytes||8+n+b>size)throw Error('bad tensor byte length');
                const t={name,fd,file:full,shape:v.shape,dtype:v.dtype,bytes,offset:8+n+a};entries.push(t);tensors.set(name,t);
            }
            for(const t of entries.sort((a,b)=>a.offset-b.offset)){if(t.offset!==end)throw Error('overlapping or non-contiguous safetensors data');end=t.offset+t.bytes;}
            if(end!==size)throw Error('unaccounted safetensors bytes');
        }
        if(!tensors.size)throw Error('no safetensors');return {tensors,close:()=>fds.forEach(fd=>fs.closeSync(fd))};
    }catch(e){fds.forEach(fd=>fs.closeSync(fd));throw e;}
}
